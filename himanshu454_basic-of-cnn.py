import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("../input/lena-image/lena_color_256.tif" , cv2.IMREAD_GRAYSCALE)
plt.imshow(img , cmap = 'gray')
plt.show()
 
img.shape
img
## Show vertical Edges

kernel = np.array([[1,0,-1] , 
                   [1,0,-1] , 
                   [1,0,-1]])
img = cv2.filter2D(img , -1 , kernel)
plt.imshow(img , cmap = 'gray')
plt.show()
## show Horizontal Edges
kernel = np.array([[1,1,1] , 
                   [0,0,0] , 
                   [-1,-1,-1]])
img = cv2.filter2D(img , -1 , kernel)
plt.imshow(img , cmap = 'gray')
plt.show()
