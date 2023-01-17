from IPython.display import HTML
HTML('<iframe width="897" height="538" src="https://www.youtube.com/embed/OgEXa3Y__Zw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
#To install opencv
#pip install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"../input/number-plate/npl.jpg")
plt.imshow(img)
plt.show()
img.shape 
kernel = np.ones((3,3))
kernel
erosion = cv2.erode(img,kernel,iterations=3)
plt.imshow(erosion)
plt.show()
dilation = cv2.dilate(img,kernel)
plt.imshow(dilation)
plt.show()
img3 = cv2.imread(r"../input/number-plate/npl2.jpg")
plt.imshow(img3)
plt.show()
#Here we have many straches, how can we tune here 
kernel = np.ones((5,5))
noise_rem = cv2.erode(img3,kernel,iterations=1)
plt.imshow(noise_rem)
plt.title("Image after removing noise")
plt.show()
#save img,age
plt.imsave("output.jpg",noise_rem)
#mix erosion and dialtion based on position
kernel1 = np.ones((7,7))
img4 = img3.copy()
img4[:,:200] =cv2.dilate(cv2.erode(img4[:,:200],kernel1),kernel1)
plt.imshow(img4)
plt.show()
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
plt.imshow(opening)
plt.show()
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing)
plt.show()
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
plt.imshow(gradient)
plt.show()                           
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
plt.imshow(tophat)
plt.show() 
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
plt.imshow(blackhat)
plt.show() 
