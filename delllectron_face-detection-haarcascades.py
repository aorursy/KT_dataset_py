#import necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import cv2
leo = cv2.imread('../input/test-images/leonardo.jpg',1)
teamusa = cv2.imread('../input/test-images/usa.jpg',1)
plt.figure(figsize=(5,10))
leo_rgb = cv2.cvtColor(leo, cv2.COLOR_BGR2RGB)
plt.imshow(leo_rgb);
face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
def detect_face (img):
    face_img = img.copy()
    detect_img = face_cascade.detectMultiScale(face_img)
    
    #get the coordinates and draw a rectangle
    for (x,y,w,h) in detect_img:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (0,255,0), 3)
    
    return face_img
#show results
plt.figure(figsize=(5,10))
plt.title('FACE DETECTION')
result = detect_face(leo_rgb)
plt.imshow(result);
usa_rgb = cv2.cvtColor(teamusa, cv2.COLOR_BGR2RGB)
result = detect_face(usa_rgb)
plt.figure(figsize=(20,8))
plt.title('TEAM USA')
plt.imshow(result);
#Adjust the detect_face function
def detect_face (img):
    face_img = img.copy()
    detect_img = face_cascade.detectMultiScale(face_img, scaleFactor=1.3, minNeighbors=3)
    
    #get the coordinates and draw a rectangle
    for (x,y,w,h) in detect_img:
        cv2.rectangle(face_img, (x,y), (x+w, y+h), (0,255,0), 3)
    
    return face_img
# show result
usa_rgb = cv2.cvtColor(teamusa, cv2.COLOR_BGR2RGB)
result = detect_face(usa_rgb)
plt.figure(figsize=(20,8))
plt.title('ADJUSTED PARAMETERS')
plt.imshow(result);
# vid_capture = cv2.VideoCapture(0)

# while True:
#     #read the frames
#     ret, frame = vid_capture.read(0)
    
#     #detect faces
#     frame = detect_face(frame)
#     cv2.imshow('Face Detection', frame);
    
#     #press esc key to exit
#     k = cv2.waitKey(1)
#     if k==27:
#         break
        
# vid_capture.release()
# cv2.destroyAllWindows()