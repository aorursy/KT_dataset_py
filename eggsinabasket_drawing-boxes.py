# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import scipy.signal
from astropy.visualization import MinMaxInterval
import itertools as it
%matplotlib inline  
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os

print(os.listdir("../input"))
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'

train_ids = os.listdir(TRAIN_PATH)

# Loading 

imgs = []
masks = []

for i in train_ids:
    img = cv2.cvtColor(cv2.imread((TRAIN_PATH + i + '/images/' + i +'.png'), -1), cv2.COLOR_BGR2RGB) #bgr 
    imgs.append(img) 

    
## plotting all images    
plt.figure(figsize=(25,50))    
#for i in range(0,100):
    #plt.subplot(20,5,i+1)
    #plt.imshow(imgs[i])
## Combining masks

def imgMerge(imgPath):
    #getting all the png files
    png_files= [f for f in os.listdir(imgPath) if f.endswith('.png')]
    
    #loading in the first image as greyscale
    img = cv2.imread(imgPath+'/'+png_files[0],0)
    for i in png_files[1:]:
        temp_img = cv2.imread(imgPath+'/'+i,0)
        img=img+temp_img
    
    return img

#merging all masks and storing inside an array
for i in range(0,len(imgs)):
    path = TRAIN_PATH + train_ids[i] + '/masks/'
    a = imgMerge(path)
    masks.append(a)
    
## checking a few side by side
plt.figure(figsize=(10, 10), dpi=100)
plt.subplot(1,2,1)
plt.imshow(masks[45])
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(imgs[45])
plt.axis('off')
## Copying image array to perform countouring
j = [np.copy(f) for f in imgs ]
mask_dup = [np.copy(f) for f in masks ]
idx = 0  ## trying out sample index

# Normalizing all images
k = [np.copy(f) for f in imgs ]

intensities = {}
for i in range(0,len(imgs)):
    img_min = np.min(imgs[i])
    img_max = np.max(imgs[i])
    k[i] = np.round(((imgs[i] - img_min)/(img_max - img_min))*255)
    k[i] = k[i].astype(np.uint8)
    d = {'Min':img_min,'Max':img_max, 'ID':i}
    intensities.update({i:d})
    
plt.imshow(k[idx])
## Trying auto canny

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged


# apply Canny edge detection using a wide threshold, tight
# threshold, and automatically determined threshold
gray = cv2.cvtColor(j[0], cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)


wide = cv2.Canny(blurred, 10, 200)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)
 
# show the images
plt.imshow(imgs[0])
plt.imshow(wide)
plt.imshow(tight)
plt.imshow(auto)

## setting manual index
idx = 0

plt.imshow(imgs[idx])
## runing canny edge detector
thresh = cv2.Canny(cv2.cvtColor(k[idx],cv2.COLOR_RGB2GRAY) ,75,75)
## runing canny edge detector
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# drawing contours
#plt.imshow(cv2.drawContours(j[100], contours, -1, (0,255,0), 2)   )

# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(j[idx], (x, y), (x+w, y+h), (0, 255, 0), 2)
 
    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
   
plt.imshow(j[idx]) 



## creating edge convolutor to extract mask contours

e_con = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

img_con = cv2.filter2D(mask_dup[idx],-1, e_con)
img_con2 = scipy.signal.convolve(mask_dup[idx], e_con, mode = 'same')

thresh1 = cv2.Canny(mask_dup[idx],50,55)
## runing canny edge detector
im2, contours, hierarchy = cv2.findContours(mask_dup[idx], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(mask_dup[idx], (x, y), (x+w, y+h), (0, 255, 0), 2)
 
    # get the min area rect
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    # convert all coordinates floating point values to int
    box = np.int0(box)
   
plt.imshow(mask_dup[idx]) 
### Creating box files for all images

boxes = {}
cont = {}
for i in range(0,len(k)):
    thresh = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(k[i],cv2.COLOR_RGB2GRAY), (3, 3), 0),75,250)
    ## runing canny edge detector
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    d = []
    cont.update({i:contours})
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        l1 = [x,y,w,h]
        d.append(l1)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(j[i], (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    boxes.update({i:d})

    
    
## checking a few side by side
plt.figure(figsize=(10, 10), dpi=100)
plt.subplot(2,2,1)
plt.imshow(k[idx]) 
plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(j[idx]) 
plt.axis('off')
plt.subplot(2,2,3)
plt.imshow(masks[idx]) 
plt.axis('off')
plt.subplot(2,2,4)
plt.imshow(mask_dup[idx]) 
plt.axis('off')
plt.tight_layout()
   
#plt.hist(masks[idx].ravel(),256,[0,256]); plt.show()


## finding all pixel addresses in a bounding box for 1 image
idx = 0
#key = 0
box_local = boxes.get(idx)
x = box_local
## get all pixel values for first object in x
x[0]
l = []
for ele in x:
    for i in range(0,ele[2]+1):
        for id in range(0,ele[3]+1):
            a1 = (ele[0]+i,ele[1]+id)
            if a1 in l:
                continue
            else:
                l.append(a1)

## check

chk = []
for t in l:
      x = t[1]
      y = t[0]
      chk.append(masks[0][x,y])
   


p = [np.copy(f) for f in masks ]

for i in range(0,len(l)):
        cv2.circle(p[0], (l[i][0], l[i][1]), 1, 255, 1)

plt.hist(k[0].ravel(),256,[0,256])

plt.subplot(1,2,1)
plt.imshow(p[0])

plt.subplot(1,2,2)
plt.imshow(masks[0])




## Finding bounding box pixel values for all images

bb_pixels = {}

for counter in range(0,len(boxes)):
    l = []
    x = boxes[counter]
    for ele in x:
        for i in range(0,ele[2]+1):
            for id in range(0,ele[3]+1):
                a1 = (ele[0]+i,ele[1]+id)
                if a1 in l:
                    continue
                else:
                    l.append(a1)
        
    bb_pixels.update({counter:l})


## Finding bounding box pixel values for all images

bb_pixels = {}

for counter in range(0,len(boxes)):
    p = []
    x = boxes[counter]
    for ele in x:
        a1 = list(it.product(range(ele[0],ele[0]+ele[2]), range(ele[1],ele[1]+ele[3])))
        p.append(a1)
    p = list(it.chain.from_iterable(p))
    p = list(set(p))    
    bb_pixels.update({counter:p})

## Now that we have the pixel values , compare against the masks to see how much we have captured and plot a histogram of capture rate
mask_pixels = {}
for key in bb_pixels.keys():
    chk = []
    x = bb_pixels[key]
    for t in x:
      x = t[1]
      y = t[0]
      chk.append(masks[key][x,y])
    mask_pixels.update({key:chk})
    
acc = []
for i in range(0,len(masks)):
        acc.append(np.sum(mask_pixels[i])/np.sum(masks[i]))
        
