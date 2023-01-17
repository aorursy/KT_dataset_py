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
import matplotlib.pyplot as plt

import numpy as np

import cv2

b=cv2.imread('../input/natural-images/natural_images/flower/flower_0002.jpg')

b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))

plt.subplot(2,2,1)

plt.imshow(b)

k=np.array([[-1,-1,-1],

            [-1,9,-1],

            [-1,-1,-1]])

sh=cv2.filter2D(b,-1,k)

plt.subplot(2,2,2)

plt.imshow(sh)

plt.show()
import matplotlib.pyplot as plt

import numpy as np

import cv2

b=cv2.imread('../input/natural-images/natural_images/flower/flower_0002.jpg')

b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))

plt.subplot(2,2,1)

plt.imshow(b)

r,sh=cv2.threshold(b,127,255,cv2.THRESH_BINARY)

plt.subplot(2,2,2)

plt.imshow(sh)

plt.show()

k=np.ones((5,5),np.uint8)

b=cv2.imread('../input/natural-images/natural_images/flower/flower_0002.jpg')

b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))

c=cv2.erode(b,k,iterations=1)

plt.subplot(3,3,1)

plt.imshow(c)

d=cv2.dilate(b,k,iterations=1)

plt.subplot(3,3,2)

plt.imshow(d)

e=cv2.morphologyEx(b,cv2.MORPH_OPEN,k)

plt.subplot(3,3,3)

plt.imshow(e)

f=cv2.morphologyEx(b,cv2.MORPH_CLOSE,k)

plt.subplot(3,3,4)

plt.imshow(f)





b=cv2.imread('../input/natural-images/natural_images/flower/flower_0002.jpg')

b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))

c=cv2.Sobel(b,cv2.CV_64F,0,1,ksize=5)

plt.subplot(3,3,1)

plt.imshow(c)



d=cv2.Sobel(b,cv2.CV_64F,0,1,ksize=5)

plt.subplot(3,3,2)

plt.imshow(d)



e=cv2.Laplacian(b,cv2.CV_64F)

plt.subplot(3,3,3)

plt.imshow(e)
b=cv2.imread('../input/natural-images/natural_images/flower/flower_0002.jpg')

b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))

c=cv2.pyrDown(b)

plt.subplot(3,3,1)

plt.imshow(c)

d=cv2.pyrUp(b)

plt.subplot(3,3,2)

plt.imshow(d)

b=cv2.imread('../input/natural-images/natural_images/flower/flower_0002.jpg')

plt.subplot(2,2,1)

plt.imshow(b)

h,w=b.shape[:2]

sr,sc=int(h*0.3),int(w*0.3)

er,ec=int(h*0.6),int(w*0.8)

bcrop=b[sr:er,sc:ec]

plt.subplot(2,2,2)

plt.imshow(bcrop)
k_nn=np.ones((12,12),np.float32)/144

blur=cv2.filter2D(b,-1,k_nn)

plt.figure(figsize=(20,20))

plt.subplot(2,2,1)

plt.imshow(b)

plt.subplot(2,2,2)

plt.imshow(blur)
b=cv2.imread('../input/natural-images/natural_images/fruit/fruit_0006.jpg')

bGRAY=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

edge=cv2.Canny(bGRAY,30,200)

plt.imshow(edge)
b=cv2.imread('../input/natural-images/natural_images/fruit/fruit_0006.jpg')

bGRAY=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

edge=cv2.Canny(bGRAY,30,200)

cont,hier=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

d=cv2.drawContours(b,cont,-1,(0,255,0),3)

plt.imshow(d)
b=cv2.imread('../input/natural-images/natural_images/fruit/fruit_0006.jpg')

bGRAY=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

edge=cv2.Canny(bGRAY,30,200)

cont,hier=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for c in cont:

    x,y,w,h=cv2.boundingRect(c)

    cv2.rectangle(b,(x,y),(x+w,y+h),(0,255,0),3)

plt.imshow(b)    
b=cv2.imread('../input/natural-images/natural_images/car/car_0005.jpg')

bGRAY=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

edge=cv2.Canny(bGRAY,30,200)

cont,hier=cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for c in cont:

    hull=cv2.convexHull(c)

    cv2.drawContours(b,[hull],0,(255,5,5),3)

plt.imshow(b)    

    
b=cv2.imread('../input/sampleimages/data/sky1.jpg')

bgray=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)

bgray=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)

bgray=np.float32(bgray)

hc=cv2.cornerHarris(bgray,3,3,0.05)

kernel = np.ones((7,7),np.uint8)

hc = cv2.dilate(hc, kernel, iterations = 10)

b[hc>0.025*hc.max()]=[255,0,255]

plt.figure(figsize=(10,10))

plt.imshow(b)
image = cv2.imread('../input/opencv-samples-images/data/sudoku.png')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(20, 20))



# Grayscale and Canny Edges extracted

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 100, 170, apertureSize = 3)



plt.subplot(2, 2, 1)

plt.title("edges")

plt.imshow(edges)



# Run HoughLines using a rho accuracy of 1 pixel

# theta accuracy of np.pi / 180 which is 1 degree

# Our line threshold is set to 240 (number of points on line)

lines = cv2.HoughLines(edges, 1, np.pi/180, 200)



# We iterate through each line and convert it to the format

# required by cv.lines (i.e. requiring end points)

for line in lines:

    rho, theta = line[0]

    a = np.cos(theta)

    b = np.sin(theta)

    x0 = a * rho

    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))

    y1 = int(y0 + 1000 * (a))

    x2 = int(x0 - 1000 * (-b))

    y2 = int(y0 - 1000 * (a))

    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)





plt.subplot(2, 2, 2)

plt.title("Hough Lines")

plt.imshow(image)

import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

img = cv2.imread('../input/sampleimages/data/sky1.jpg')

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.figure(figsize=(20,20))

plt.subplot(121),plt.imshow(img)

plt.subplot(122),plt.imshow(dst)

plt.show()

import numpy as np

import matplotlib.pyplot as plt

import cv2

img = cv2.imread('../input/sampleimages/data/sky1.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(20,20))

plt.subplot(1,2,1),plt.imshow(img)



Z = img.reshape((-1,3))

Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 8	

ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)

res = center[label]

res2 = res.reshape((img.shape))

plt.subplot(1,2,2),plt.imshow(res2)

import numpy as np

import cv2 as cv

from matplotlib import pyplot as plt

imgL = cv2.imread('../input/opencv-samples-images/data/left01.jpg',0)

imgR = cv2.imread('../input/opencv-samples-images/data/right01.jpg',0)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)

disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')

plt.show()

b = cv2.imread('../input/opencv-samples-images/data/left01.jpg')

c = cv2.imread('../input/opencv-samples-images/data/left01.jpg')

d=cv2.addWeighted(b,0,c,0.2,0)

plt.figure(figsize=(20,20))

plt.subplot(221)

plt.imshow(d)

plt.subplot(222)

b=cv2.addWeighted(b,0.6,c,0,0)



plt.imshow(b)
import cv2

import numpy as np

import math

from vcam import vcam,meshGen

import matplotlib.pyplot

plt.figure(figsize=(20,20))

img=cv2.imread("../input/opencv-samples-images/data/right03.jpg")

H,W=img.shape[:2]

cam1=vcam(H=H,W=W)

p1=meshGen(H,W)

p1.Z += 20*np.exp(-0.5*((p1.X*1.0/p1.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))

#p1.Z += 20*np.exp(-0.5*((p1.Y*1.0/p1.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))

#plane.Z += 20*np.exp(-0.2*((p1.Y*1.0/p1.W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))

pt3d=p1.getPlane()

pt2d=cam1.project(pt3d)

mp_x,mp_y=cam1.getMaps(pt2d)

output=cv2.remap(img,mp_x,mp_y,interpolation=cv2.INTER_LINEAR)

plt.subplot(1,2,1)

plt.imshow(cv2.cvtColor(np.hstack((img,output)),cv2.COLOR_BGR2RGB))





!pip install vcam
import cv2 

import matplotlib.pyplot as plt



algo = 'MOG2'



if algo == 'MOG2':

    backSub = cv2.createBackgroundSubtractorMOG2()

else:

    backSub = cv2.createBackgroundSubtractorKNN()



plt.figure(figsize=(20, 20))



frame = cv2.imread('../input/natural-images/natural_images/motorbike/motorbike_0007.jpg')

fgMask = backSub.apply(frame)



plt.subplot(2, 2, 1)

plt.title("Frame")

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



plt.subplot(2, 2, 2)

plt.title("FG Mask")

plt.imshow(cv2.cvtColor(fgMask, cv2.COLOR_BGR2RGB))
image = cv2.imread('/kaggle/input/opencv-samples-images/WaldoBeach.jpg')

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



plt.figure(figsize=(30, 30))



plt.subplot(2, 2, 1)

plt.title("Where is Waldo?")

plt.imshow(image)



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# Load Template image

template = cv2.imread('/kaggle/input/opencv-samples-images/waldo.jpg',0)



result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)



#Create Bounding Box

top_left = max_loc

bottom_right = (top_left[0] + 50, top_left[1] + 50)

cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)



plt.subplot(2, 2, 2)

plt.title("Waldo")

plt.imshow(image)
img=cv2.imread('../input/natural-images/natural_images/flower/flower_0013.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)

th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)

plt.subplot(2,2,1)	

#thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5) 

plt.imshow(th2)

plt.subplot(2,2,2)

plt.imshow(th3)

plt.show()

img=cv2.imread('../input/natural-images/natural_images/flower/flower_0013.jpg')

plt.figure(figsize=(20,20))

plt.subplot(3,3,1)

plt.imshow(img)

b=cv2.GaussianBlur(img,(15,15),0)



plt.subplot(3,3,2)

plt.imshow(b)

c=cv2.medianBlur(img,5)

plt.subplot(3,3,3)

plt.imshow(c)

#d=cv2.bilateralFilter(img,9,75,75)

#plt.subplot(3,3,4)

#plt.imshow(d)
img = cv2.imread('../input/natural-images/natural_images/flower/flower_0001.jpg')

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.figure(figsize=(25,25))

plt.subplot(2,2,1)

plt.imshow(img)

h,w = img.shape[:2]

qh,qw=h/4,w/4

M = np.float32([[1,0,qh],[0,1,qw]])

dst = cv2.warpAffine(img,M,(h,w))

plt.subplot(2,2,2)

plt.imshow(dst)

plt.figure(figsize=(25,25))

plt.subplot(2,2,1)

plt.imshow(img)

h,w = img.shape[:2]

qh,qw=h/4,w/4

M = cv2.getRotationMatrix2D((qw,qh),90,1)

dst = cv2.warpAffine(img,M,(h,w))

plt.subplot(2,2,2)

plt.imshow(dst) 
