import cv2 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
face_cascade = cv2.CascadeClassifier('../input/opencv-haarcascade/data/haarcascades/haarcascade_frontalface_default.xml')
gray =cv2.imread( "../input/opencv-samples-images/data/lena.jpg",0)
plt.figure(figsize=(10, 10))
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    ax = plt.gca()
    ax.add_patch( Rectangle((x,y), 
                       w,   h,
                        fc ='none',  
                        ec ='b', 
                        lw = 4) ) 

plt.imshow(gray,cmap = 'gray')
plt.title('template'), plt.xticks([]), plt.yticks([])

plt.show()