import cv2

import numpy as np

import matplotlib.pyplot as plt

import glob

import os

from tqdm import tqdm



%matplotlib inline
X=[]

Z=[]

IMG_SIZE=150

FLOWER_SUNFLOWER_DIR='../input/flowers-recognition/flowers/flowers/sunflower'
X=[]

Z=[]

IMG_SIZE=150

FLOWER_SUNFLOWER_DIR='../input/flowers-recognition/flowers/flowers/sunflower'

def assign_label(img,flower_type):

    return flower_type

#FUNCTION TO LOAD DATA

def make_train_data(flower_type,DIR):

    for img in tqdm(os.listdir(DIR)):

        label=assign_label(img,flower_type)

        path = os.path.join(DIR,img)

        img = cv2.imread(path,cv2.IMREAD_COLOR)

        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))#Resizing the image

        

        X.append(np.array(img))#we are storing the data in the form of a list

        Z.append(str(label))# we are also storing the label in the form of a list
#Loading Sunflower Data

make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)

print(len(X))
fix_img = cv2.cvtColor(X[0],cv2.COLOR_BGR2RGB)# THIS IS HOW TO CONVERT BGR COLOR SPACE TO RGB COLOR SPACE
plt.figure(figsize = (12,8))

plt.imshow(fix_img)
new_img_1 = fix_img.copy() 

new_img_1.shape

new_img_1[:,:,0] = 0 # making R channel zero

new_img_1[:,:,1] = 0 #making G channel zero
plt.imshow(new_img_1) # Finally having blue version of that image
new_img_2 = fix_img.copy()

new_img_3 = fix_img.copy()
#For Red color Channel

new_img_2[:,:,1] = 0

new_img_2[:,:,2] = 0

#For Green color channel

new_img_3[:,:,0] = 0

new_img_3[:,:,2] = 0
f, axes = plt.subplots(1,3, figsize = (15,15))

list = [new_img_1,new_img_2,new_img_3]

i = 0

for ax in axes:

    ax.imshow(list[i])

    i+=1
f, axes = plt.subplots(1,3, figsize = (15,15))

list = [fix_img[:,:,0],fix_img[:,:,1],fix_img[:,:,2]]

i = 0

for ax in axes:

    ax.imshow(list[i],cmap = 'gray')

    i+=1
hsl_img = cv2.cvtColor(X[0],cv2.COLOR_BGR2HLS)

hsl_img.shape

plt.figure(figsize=(12,10))

plt.imshow(hsl_img)
hsl_img_1 = hsl_img.copy()

hsl_img_2 = hsl_img.copy()

hsl_img_3 = hsl_img.copy()
#HUE --> ZERO

hsl_img_1[:,:,1] = 0

hsl_img_1[:,:,2] = 0

#SATURATION --> ZERO

hsl_img_2[:,:,0] = 0

hsl_img_2[:,:,2] = 0

#LIGHTNESS --> ZERO

hsl_img_3[:,:,0] = 0

hsl_img_3[:,:,1] = 0
f, axes = plt.subplots(1,3, figsize = (15,15))

list = [hsl_img_1,hsl_img_2,hsl_img_3]

i = 0

for ax in axes:

    ax.imshow(list[i])

    i+=1
f, axes = plt.subplots(1,3, figsize = (15,15))

list = [hsl_img[:,:,0],hsl_img[:,:,1],hsl_img[:,:,2]]

i = 0

for ax in axes:

    ax.imshow(list[i],cmap = "gray")

    i+=1
hsv_img = cv2.cvtColor(X[0],cv2.COLOR_BGR2HSV)
hsv_img.shape
plt.figure(figsize = (10,8))

plt.imshow(hsv_img)
hsv_img_1 = hsv_img.copy()

hsv_img_2 = hsv_img.copy()

hsv_img_3 = hsv_img.copy()
f, axes = plt.subplots(1,3, figsize = (15,15))

list = [hsv_img_1,hsv_img_2,hsv_img_3]

i = 0

for ax in axes:

    ax.imshow(list[i])

    i+=1
f, axes = plt.subplots(1,3, figsize = (15,15))

list = [hsv_img[:,:,0],hsv_img[:,:,1],hsv_img[:,:,2]]

i = 0

for ax in axes:

    ax.imshow(list[i],cmap = "gray")

    i+=1
!pip install opencv-contrib-python==4.2.0.34
import cv2

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

pic = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/leaf.jfif')

pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)

plt.imshow(pic)
type(pic)
matrix = np.float32([[1,0,20],[0,1,30]])
rows,cols,chn = pic.shape

print('height:{}'.format(rows))

print('width:{}'.format(cols))

print("pic shape: {} ".format(pic.shape))
translated_pic = cv2.warpAffine(pic,matrix,(cols,rows))
plt.figure(figsize=(10,8))

plt.subplot(121)

plt.imshow(translated_pic)



# plt.figure(figsize=(10,8))

plt.subplot(122)

plt.imshow(pic)

plt.show()
translated_pic.shape
pic = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/leaf.jfif')

pic = cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)

plt.imshow(pic)
pic.shape
resized_pic = cv2.resize(pic,(100,200))

plt.imshow(resized_pic)
resized_pic.shape
width_ratio = 0.5

height_ratio = 0.5

resized_pic2 = cv2.resize(pic,(0,0),pic,width_ratio,height_ratio)
plt.imshow(resized_pic2)

print('shape:{}'.format(resized_pic2.shape))# shape reduced by 50%
rotate = cv2.flip(pic,0)#you also can pass value like--> 0,1,-1 etc.
plt.imshow(rotate)# Rotated by 180 degree
pic1 = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/gorilla.jpg')

pic1 = cv2.cvtColor(pic1,cv2.COLOR_BGR2RGB)
plt.imshow(pic1)
pic1.shape
crop = pic1[800:3000,1800:3200]
plt.imshow(crop)#Croped Image
import cv2

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
img = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/SharedScreenshot1.jpg',0)

plt.imshow(img,cmap = 'gray')
img.max()/2#Finding middle value of the distribution of the pixel value
'''1st arg as the gray image itself the 2nd arg as the threshold value(usually the mean value of pixel values) then 3rd 

arg as the the max pixel value and as the last arg we pass the method using which we are going to do the thresholding'''

ret3,thresh3 = cv2.threshold(img,123,255,cv2.THRESH_TRUNC)
ret, thresh1 = cv2.threshold(img,123,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img,123,255,cv2.THRESH_TRIANGLE)
plt.imshow(thresh1,cmap = 'gray')
plt.imshow(thresh2,cmap = 'gray')
diff = thresh1 - thresh2
plt.imshow(diff)
diff1 = thresh1 - thresh3
sum1 = thresh1 + thresh3
plt.imshow(sum1,cmap = 'gray')
'''1st arg: Gray image itself

   2nd arg:Max pixel value

   3rd arg: Type for calculating mean

   4th arg:thresholding type

   5th arg: block size(size of the pixel neighbourhood for calculate a threshold, it is needed to be odd like-3,5,7 etc.)

   6th arg: c constant(Generally a constant value & a odd number)'''

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,8)
th = th3 - th2

th.shape
'''import numpy as np

th1 = np.mean(th)

print(th1)'''

plt.imshow(th,cmap = 'gray')
#Function for Loading the image

def load_img():

    img = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/blue_brick.jpg')

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    return img
#Function for displaying the image

def disp_img(img,cmap = None):

    fig = plt.figure(figsize = (8,8))

    ax = fig.add_subplot(111)

    ax.imshow(img,cmap)
img1 = load_img()
img1.shape
disp_img(img1)
img2 = load_img()

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img2,text = 'Bricks',org = (10,125),fontFace = font,fontScale = 2,color = (255,0,0),thickness = 1)
disp_img(img2)
#Creating a custom kernel as a filter for our bluring perpous

import numpy as np

kernel = np.ones(shape=(3,3),dtype = np.float32)/6.07 ### WE ARE DIVIDING THAT BY 25(KERNEL SIZE= 5X5)

#TO GET FLOTING VALUES IF WE USE THAT FOR MULTIPLYING THE OTHER IMAGE THEN PIXEL VALUE WILL BE DECREASED OF OTHER IMAGE
1/25
dst = cv2.filter2D(img2,-1,kernel)

disp_img(dst)
#Loading image again 

img2 = load_img()

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img2,text = 'Bricks',org = (10,125),fontFace = font,fontScale = 2,color = (255,0,0),thickness = 1)

print('reset')
disp_img(img2)
blur = cv2.blur(img2,ksize = (2,2))

disp_img(blur)
noisy_img2  = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/balloons_noisy.png')

noisy_img2 = cv2.cvtColor(noisy_img2,cv2.COLOR_BGR2RGB)

disp_img(noisy_img2)

blur = cv2.blur(noisy_img2,ksize = (3,3))

disp_img(blur)
img2 = load_img()

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img2,text = 'Bricks',org = (10,125),fontFace = font,fontScale = 2,color = (255,0,0),thickness = 1)

print('reset')
disp_img(img2)
blur_img  = cv2.GaussianBlur(img2,(3,3),0.64)

disp_img(blur_img)
noisy_img1  = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/balloons_noisy.png')

noisy_img1 = cv2.cvtColor(noisy_img1,cv2.COLOR_BGR2RGB)

disp_img(noisy_img1)

blur_img  = cv2.GaussianBlur(noisy_img1,(3,3),0.64)

disp_img(blur_img)
#Gaussian blur bluring the image but not removing noise from this picture.
img2 = load_img()

font = cv2.FONT_HERSHEY_COMPLEX

cv2.putText(img2,text = 'Bricks',org = (10,125),fontFace = font,fontScale = 2,color = (255,0,0),thickness = 1)

print('reset')
disp_img(img2)
median_blur = cv2.medianBlur(img2,3)

disp_img(median_blur)
noisy_img  = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/balloons_noisy.png')

noisy_img = cv2.cvtColor(noisy_img,cv2.COLOR_BGR2RGB)

disp_img(noisy_img)

median = cv2.medianBlur(noisy_img,5)

disp_img(median)
plt.figure(figsize=(12,10))

attatiched = np.concatenate([noisy_img,median],axis=1)

plt.imshow(attatiched)
#READING THREE DIFFERENT IMAGES

dark =cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/dark_image.jpg')

dark_img = cv2.cvtColor(dark,cv2.COLOR_BGR2RGB)



rainbo =cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/rainbo.jpg')

rainbo_img = cv2.cvtColor(rainbo,cv2.COLOR_BGR2RGB)



brick =cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/blue_brick.jpg')

brick_img = cv2.cvtColor(brick,cv2.COLOR_BGR2RGB)
#SHOWING THE IMAGE

disp_img(brick_img)
rainbo_hist = cv2.calcHist([rainbo],channels = [0],mask = None,histSize= [256],ranges = [0,256])
rainbo_hist.shape
plt.plot(rainbo_hist)

#plt.xlim([0,20])
color = ('b','g','r')



for i,col in enumerate(color):

    hist = cv2.calcHist([brick],[i],None,[256],[0,256])

    plt.plot(hist,color = col)

    plt.xlim([0,256])

    #plt.ylim()

plt.title('hist for rainbo')
rainbo =cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/rainbo.jpg')

rainbo_img = cv2.cvtColor(rainbo,cv2.COLOR_BGR2RGB)
rainbo_img.shape
plt.imshow(rainbo_img)
mask = np.zeros(rainbo_img.shape[:2],np.uint8)
plt.imshow(mask,cmap = 'gray')
mask[75:100,50:100] = 255
plt.imshow(mask,cmap = 'gray')
masked_img = cv2.bitwise_and(rainbo,rainbo,mask = mask)

show_masked_img = cv2.bitwise_and(rainbo_img,rainbo_img,mask = mask)
plt.imshow(show_masked_img)
hist_mask = cv2.calcHist([rainbo],[2],mask,[256],[0,256])

#Notice we are only passing "[2]" in the number of color 

#channel which reffers that here we are only interested about "RED" color channel, as in the maskind image we only have

#green color not red so as output we will not have any high picks in the histogram
hist_not_mask = cv2.calcHist([rainbo],[2],None,[256],[0,256])#Performing same thing without mask
plt.plot(hist_mask)
hist_mask_green = cv2.calcHist([rainbo],[1],mask,[256],[0,256])#trying to plot histogram for only green color channel
plt.plot(hist_mask_green)
gorilla = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/gorilla.jpg')

gorilla = cv2.cvtColor(gorilla,cv2.COLOR_BGR2GRAY)

disp_img(gorilla,cmap = 'gray')
eq_gorilla = cv2.equalizeHist(gorilla)
disp_img(eq_gorilla,cmap = 'gray')#After applying histogram equalization
road = cv2.imread('../input/cv-object-detecton-dataset/CV Object Detecton Dataset/yello_road.jpg')

road = cv2.cvtColor(road,cv2.COLOR_BGR2RGB)

plt.imshow(road)
blurry = cv2.GaussianBlur(road,(35,35),0)

plt.imshow(blurry)

blur_img = blurry.copy()
import numpy as np
blur_img4 = blurry.copy()
hsv = cv2.cvtColor(blur_img4,cv2.COLOR_RGB2HSV)

low_yellow = np.array([18,94,140])

up_yellow = np.array([48,230,230])

mask = cv2.inRange(hsv,low_yellow,up_yellow)

edge = cv2.Canny(mask,75,150)



line = cv2.HoughLinesP(edge,1,np.pi/180,50,maxLineGap = 50)



for i in line:

    x1,y1,x2,y2 = i[0]

    cv2.line(blur_img4,(x1,y1),(x2,y2),(0,255,0),25)

    

plt.imshow(blur_img4)