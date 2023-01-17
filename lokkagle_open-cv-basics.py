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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
path = '/kaggle/input/cat.2012.jpg'
img1 = cv2.imread(path, 1) # 1 = colored image
img2 = cv2.imread(path, 0) # 0 = gray image
img1[:1]
img2
print('shape of img1: {}'.format(img1.shape))
print('3 for 3 channels.i.e. red, blue, green', '\n')
print('shape of img2: {}'.format(img2.shape))
# show images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
fig.suptitle('color   vs    grayscale')
ax1.imshow(img1)
ax2.imshow(img2, cmap = 'gray')
plt.show()
# resizing
resize1 = cv2.resize(img1, (200,300))
plt.imshow(resize1, cmap= 'gray')
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5))
fig.suptitle('color   vs    grayscale')
ax1.imshow(cv2.resize(img1, (200,300)))
ax2.imshow(cv2.resize(img2,(800,800)), cmap = 'gray')
plt.show()
# increasing it's size by double  
resize_image1 = cv2.resize(img1, (int(img1.shape[0]*2),int(img1.shape[1]*2)))
print('original shape of img1 :{}'.format(img1.shape))
plt.imshow(img1)
plt.show()
print('after resized : {}'.format(resize_image1.shape))
plt.imshow(resize_image1)
plt.show()
# grayscale 
img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.imshow(img_gray)
plt.show()
# gaussain blurr
img_gblur = cv2.GaussianBlur(img1, (7,7), 50)
plt.imshow(img_gblur)
plt.show()
# canny or edge detection
img_canny = cv2.Canny(img1, threshold1 = 100, threshold2 = 100)
plt.imshow(img_canny)
plt.show()
img_canny = cv2.Canny(img1, threshold1 = 200, threshold2 = 200)
plt.imshow(img_canny)
plt.show()
# image dialation
# img dialation adds the thickness to the edge images with lighter edges
kernal = np.ones((5,5), np.uint8)
imgdialation1 = cv2.dilate(img_canny, kernal, iterations = 1 ) # if we increase the thickness then iterations has to change
plt.imshow(imgdialation1)
plt.show()
kernal = np.ones((5,5), np.uint8)
imgdialation2 = cv2.dilate(img_canny, kernal, iterations = 3 ) # if we increase the thickness then iterations has to change
plt.imshow(imgdialation2)
plt.show()
# image erode
img_eroded = cv2.erode(imgdialation1, kernal, iterations = 1) # it will thicken the image
plt.imshow(img_eroded)
plt.show()
# scales of an image
plt.figure(figsize=(15,10)) # specifying the overall grid size

img_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img_gblur = cv2.GaussianBlur(img1, (7,7), 50)
img_canny = cv2.Canny(img1, threshold1 = 100, threshold2 = 100)
kernal = np.ones((5,5), np.uint8)
imgdialation1 = cv2.dilate(img_canny, kernal, iterations = 1 )
imgdialation2 = cv2.dilate(img_canny, kernal, iterations = 3 )
img_eroded = cv2.erode(imgdialation1, kernal, iterations = 1) 

the_array = [img_gray,img_gblur, img_canny,  imgdialation1,imgdialation2, img_eroded]

for i in range(6):
    plt.subplot(2,3,i+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(the_array[i])
plt.tight_layout()
plt.show()

plt.imshow(cv2.line(img1, (0,0), (300,300), (255,0,0), 4))
plt.imshow(cv2.line(img1, (100,10), (300,300), (0,255,0), 4))
plt.imshow(cv2.line(img1, (200,0), (200,300), (0,0,255), 4))
plt.imshow(cv2.rectangle(img1, (50,50), (300,300), (0,255,0), 4))
plt.imshow(cv2.circle(img1, (400,150), 100, (255,0,0),3))
plt.imshow(cv2.putText(img1, "Hey! Cat", (320,300), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),3))
plt.show()
import cv2
# # create a cascadeclassifier object
# path = 'kaggle/input/cat.2012.jpg'
# path1 = "kaggle/input/haarcascades/haarcascade_frontalcatface.xml"
# face_cascade = cv2.CascadeClassifier(path1)
# img = cv2.imread(path, 0) # 0 means gray scale image
# grey_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(grey_image, scaleFactor = 1.05, minNeighbors = 2)
# for x,y,w,h in faces:
#     img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
#     plt.imshow(img)
#     plt.show()