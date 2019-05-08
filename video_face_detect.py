import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
plt.rcParams['figure.figsize'] = (224, 224)

face_cascade = cv2.CascadeClassifier('modelo/haarcascade_frontalface_default.xml')
vc = cv2.VideoCapture('video/Can You Watch This Without Smiling.mp4')
cv2.namedWindow('Can You Watch This Without Smiling',cv2.WINDOW_AUTOSIZE)

largura_video = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
altura_video = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_faces_detectadas = cv2.VideoWriter('saida/video_faces_detectadas.mp4',cv2.VideoWriter_fourcc('X','2','6','4'), 10, (largura_video,altura_video))

while vc.isOpened():

    ret, img = vc.read()

    if ret == True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

        cv2.imshow('Can You Watch This Without Smiling',img)
        time.sleep(0.01)
        video_faces_detectadas.write(img)
        
    else:
        break

    if cv2.waitKey(1) == 27:
            break  # esc para encerrar vídeo        

vc.release()
video_faces_detectadas.release()
cv2.destroyAllWindows()