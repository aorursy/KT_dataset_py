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
import cv2

import numpy as np

from matplotlib import pyplot as plt



img = cv2.imread('/kaggle/input/small-egg-count/30.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(gray)

plt.imshow(v, 'gray')

plt.show()
blur = cv2.GaussianBlur(v, (3,3), 0)

plt.imshow(v, 'gray')

plt.show()
ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)



plt.subplot(1, 3, 1), plt.imshow(blur, 'gray')

plt.title('Gaussian filtered image'), plt.xticks([]), plt.yticks([])



plt.subplot(1,3,2), plt.hist(blur.ravel(), 256)

plt.title('Histogram'), plt.xticks([]), plt.yticks([])

plt.subplot(1,3,3),plt.imshow(th,'gray')

plt.title('Otsu Thresholding'), plt.xticks([]), plt.yticks([])

plt.show()

 

cv2.imwrite('otsuThesholding.jpg', th)
sobel = cv2.Sobel(th, cv2.CV_8UC1, 1, 1, ksize=5)



plt.subplot(1, 1, 1), plt.imshow(sobel, cmap='gray')

plt.title('sobel Edge detection'), plt.xticks([]), plt.yticks([])

plt.show()



cv2.imwrite('sobel.jpg', sobel)
gray_img = cv2.imread('sobel.jpg', 0)

cimg = cv2.cvtColor(sobel,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(gray_img,cv2.HOUGH_GRADIENT,1,20,

                            param1=45,param2=32,minRadius=0,maxRadius=0)

 

circles = np.uint16(np.around(circles))

 

for i in circles[0,:]:

    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)

    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

 

plt.subplot(122),plt.imshow(cimg)

plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])

plt.show()

 

cv2.imwrite('houghCircle.jpg', cimg)