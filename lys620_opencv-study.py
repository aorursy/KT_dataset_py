# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import matplotlib.pyplot as plt
import cv2
image = cv2.imread('/kaggle/input/opencv-samples-images/data/building.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

kernel_sharpening = np.array([[-1,-1,-1],
                             [-1,9,-1],
                             [-1,-1,-1]
                             ])

sharpened = cv2.filter2D(image, -1, kernel_sharpening) ## -1 : original size

plt.subplot(1,2,2)
plt.title("Image Sharpening")
plt.imshow(sharpened)
plt.show()
image = cv2.imread('/kaggle/input/opencv-samples-images/Origin_of_Species.jpg', 0)

plt.figure(figsize=(30,30))
plt.subplot(3,2,1)
plt.title("Original")
plt.imshow(image)

ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
plt.subplot(3,2,2)
plt.title("Threshold Binary")
plt.imshow(thresh1)
image = cv2.GaussianBlur(image,(3,3),0)
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
plt.figure(figsize=(30,30))
plt.subplot(3,2,3)
plt.title("Adaptive Mean Thresholding")
plt.imshow(thresh)
from skimage.data import coins
img = coins()
maxval = 255
thresh = maxval / 2

_,thresh1 = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)
_,thresh2 = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY_INV)
_,thresh3 = cv2.threshold(img, thresh, maxval, cv2.THRESH_TRUNC)
_,thresh4 = cv2.threshold(img, thresh, maxval, cv2.THRESH_TOZERO)
_,thresh5 = cv2.threshold(img, thresh, maxval, cv2.THRESH_TOZERO_INV)

titles = ['Original','THRESH_BINARY','THRESH_BINARY_INV','THRESH_TRUNC','THRESH_TOZERO','THRESH_TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

plt.figure(figsize=(9,5))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i], fontdict={'fontsize':10})
    plt.axis('off')
    
plt.tight_layout(pad=0.7)
plt.show()
image = cv2.imread('/kaggle/input/opencv-samples-images/data/fruits.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, channel = image.shape
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
plt.figure(figsize=(20,20))
plt.subplot(3,2,1)
plt.title("Original")
plt.imshow(image)

plt.subplot(3,2,2)
plt.title("Sobel_x")
plt.imshow(sobel_x)

plt.subplot(3,2,3)
plt.title("Sobel_y")
plt.imshow(sobel_y)

sobel_OR = cv2.bitwise_or(sobel_x,sobel_y)
plt.subplot(3,2,4)
plt.title("Sobel_OR")
plt.imshow(sobel_OR)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
plt.subplot(3,2,5)
plt.title("Laplacian")
plt.imshow(laplacian)

canny = cv2.Canny(image, 50, 120)
plt.subplot(3,2,6)
plt.title("Canny")
plt.imshow(canny)
image = cv2.imread('/kaggle/input/opencv-samples-images/scan.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image)

points_A = np.float32([[320,15],[700,215],[85,610],[530,780]])
points_B = np.float32([[0,0],[420,0],[0,594],[420,594]])

M = cv2.getPerspectiveTransform(points_A,points_B)
warped = cv2.warpPerspective(image,M,(420,594))

plt.subplot(1,2,2)
plt.title("warpPerspective")
plt.imshow(warped)
image = cv2.imread('/kaggle/input/opencv-samples-images/data/fruits.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(image)

image_scaled = cv2.resize(image,None,fx=0.75,fy=0.75)
plt.subplot(2,2,2)
plt.title("Scaling - Linear Interpolation")
plt.imshow(image_scaled)

img_scaled = cv2.resize(image,None,fx=2,fy=2,interpolation = cv2.INTER_CUBIC)
plt.subplot(2,2,3)
plt.title("Scaling - Cubic Interpolation")
plt.imshow(img_scaled)

img_scaled = cv2.resize(image,(900,400),interpolation = cv2.INTER_AREA)

plt.subplot(2,2,4)
plt.title("Scaling - Skewed Size")
plt.imshow(img_scaled)
image = cv2.imread('/kaggle/input/opencv-samples-images/data/butterfly.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20, 20))

plt.subplot(2, 2, 1)
plt.title("Original")
plt.imshow(image)

smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)

plt.subplot(2, 2, 2)
plt.title("Smaller")
plt.imshow(smaller)

plt.subplot(2, 2, 3)
plt.title("Larger")
plt.imshow(larger)
image = cv2.imread('/kaggle/input/opencv-samples-images/data/messi5.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(image)

height,width = image.shape[:2]
start_row,start_col = int(height*0.25), int(width*0.25)
end_row,end_col = int(height*0.75), int(width*0.75)
cropped = image[start_row:end_row,start_col:end_col]

plt.subplot(2,2,2)
plt.title("Cropped")
plt.imshow(cropped)
image = cv2.imread('/kaggle/input/opencv-samples-images/data/home.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.title("Original")
plt.imshow(image)

kernel_3X3 = np.ones((3,3),np.float32)/9
blurred = cv2.filter2D(image,-1,kernel_3X3)
plt.subplot(2,2,2)
plt.title("3X3 Kernel Blurring")
plt.imshow(blurred)

kernel_7X7 = np.ones((7,7),np.float32)/49
blurred2 = cv2.filter2D(image,-1,kernel_7X7)
plt.subplot(2,2,3)
plt.title("7X7 Kernel Blurring")
plt.imshow(blurred2)
image = cv2.imread('/kaggle/input/opencv-samples-images/data/pic3.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.title("Original")
plt.subplot(2,2,1)
plt.imshow(image)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray,30,200)
plt.subplot(2,2,2)
plt.title("Canny Edges")
plt.imshow(edged)

contours, hierarchy = cv2.findContourson(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
plt.subplot(2,2,3)
plt.title("Canny Edges After Contouring")
plt.imshow(edged)
print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(image, contours, -1, (0,255,0), 3)
plt.subplot(2,2,4)
plt.title("Contours")
plt.imshow(image)
