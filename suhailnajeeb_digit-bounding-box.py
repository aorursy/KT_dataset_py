# Importing necessary libraries
import numpy as np
import pandas as pd
import os
import glob
import cv2 # the opencv library
import matplotlib.pyplot as plt
FIG_WIDTH=16 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows
data_dir=os.path.join('..','input')
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))

paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))
img_gray=cv2.imread(os.path.join(data_dir,'training-a','a00000.png'),cv2.IMREAD_GRAYSCALE) # read image
plt.imshow(img_gray,cmap='gray')
plt.show()
(thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
plt.imshow(img_bin,cmap='gray')
plt.title('Threshold: {}'.format(thresh))
plt.show()
img_bin=255-img_bin
plt.imshow(img_bin,cmap='gray')
plt.show()
_,contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
plt.figure(figsize=(20,4))
for i in range(len(contours)):
    img_gray_cont=cv2.drawContours(img_gray.copy(), contours, i,255, 1)
    plt.subplot((len(contours)//4)+1,4,i+1)
    plt.title('Contour: {}'.format(i+1))
    plt.imshow(img_gray_cont,cmap='gray')
plt.show()
countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]
bb=cv2.boundingRect(countours_largest)
# pt1 and pt2 are terminal coordinates of the diagonal of the rectangle
pt1=(bb[0],bb[1]) # upper coordinates 
pt2=(bb[0]+bb[2],bb[1]+bb[3]) # lower coordinates
img_gray_bb=img_gray.copy()
cv2.rectangle(img_gray_bb,pt1,pt2,255,1)
plt.imshow(img_gray_bb,cmap='gray')
plt.show()
PIXEL_ADJ=3 # the number of padding pixels
pt1=[bb[0],bb[1]]
pt2=[bb[0]+bb[2],bb[1]+bb[3]]

if pt1[0]-PIXEL_ADJ>=0: # upper x coordinate
    pt1[0]=pt1[0]-PIXEL_ADJ
else: pt1[0]=0
if pt1[1]-PIXEL_ADJ>=0: # upper y coordinate
    pt1[1]=pt1[1]-PIXEL_ADJ
else: pt1[1]=0

if pt2[0]+PIXEL_ADJ<=img_gray.shape[0]: # lower x coordinate
    pt2[0]=pt2[0]+PIXEL_ADJ
else: pt2[0]=img_gray.shape[0]
if pt2[1]+PIXEL_ADJ<=img_gray.shape[1]: # lower y coordinate
    pt2[1]=pt2[1]+PIXEL_ADJ
else: pt2[1]=img_gray.shape[1]
pt1=tuple(pt1)
pt2=tuple(pt2)
img_gray_bb=img_gray.copy()
cv2.rectangle(img_gray_bb,pt1,pt2,255,1)
plt.imshow(img_gray_bb,cmap='gray')
plt.show()
img_gray=cv2.imread(os.path.join(data_dir,'training-a','a00061.png'),cv2.IMREAD_GRAYSCALE) # read image, image size is 180x180
plt.imshow(img_gray,cmap='gray')
plt.show()
(thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
plt.imshow(img_bin,cmap='gray')
plt.title('Threshold: {}'.format(thresh))
plt.show()
img_bin=255-img_bin
plt.imshow(img_bin,cmap='gray')
plt.show()
_,contours,_ = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
plt.figure(figsize=(20,8))
for i in range(len(contours)):
    img_gray_cont=cv2.drawContours(img_gray.copy(), contours, i,255, 1)
    plt.subplot((len(contours)//4)+1,4,i+1)
    plt.title('Contour: {}'.format(i+1))
    plt.imshow(img_gray_cont,cmap='gray')
plt.show()
kernel = np.ones((5,5),np.uint8)
img_dilate=cv2.dilate(img_bin,kernel,iterations = 1)
plt.imshow(img_dilate,cmap='gray')
plt.show()
img_erode=cv2.erode(img_dilate,kernel,iterations = 1)
plt.imshow(img_erode,cmap='gray')
plt.show()
def img_bb(path):
    # shows image with bounding box
    BB_COLOR=255 # bounding box color
    img_gray=cv2.imread(path,cv2.IMREAD_GRAYSCALE) # read image
    (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_OTSU)
    # if the background is white, invert image
    if len(img_bin[img_bin==255])>len(img_bin[img_bin==0]):
        img_bin=255-img_bin
        BB_COLOR=0
    
    kernel = np.ones((5,5),np.uint8)
    img_dilate=cv2.dilate(img_bin,kernel,iterations = 1)
    img_erode=cv2.erode(img_dilate,kernel,iterations = 1)
    
    _,contours,_ = cv2.findContours(img_erode.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]
    bb=cv2.boundingRect(countours_largest)
    
    pt1=[bb[0],bb[1]]
    pt2=[bb[0]+bb[2],bb[1]+bb[3]]
    # pt1 and pt2 are terminal coordinates of the diagonal of the rectangle
    if pt1[0]-PIXEL_ADJ>=0: # upper x coordinate
        pt1[0]=pt1[0]-PIXEL_ADJ
    else: pt1[0]=0
    if pt1[1]-PIXEL_ADJ>=0: # upper y coordinate
        pt1[1]=pt1[1]-PIXEL_ADJ
    else: pt1[1]=0
    if pt2[0]+PIXEL_ADJ<=img_gray.shape[0]: # lower x coordinate
        pt2[0]=pt2[0]+PIXEL_ADJ
    else: pt2[0]=img_gray.shape[0]
    if pt2[1]+PIXEL_ADJ<=img_gray.shape[0]: # lower y coordinate
        pt2[1]=pt2[1]+PIXEL_ADJ
    else: pt2[1]=img_gray.shape[0]
    pt1=tuple(pt1)
    pt2=tuple(pt2)
    
    img_gray_bb=img_gray.copy()
    cv2.rectangle(img_gray_bb,pt1,pt2,BB_COLOR,3)
    # this function returns the image with bounding box drawn it,
    # if you need the bounding box instead, simply make it return pt1 and pt2
    return img_gray_bb
    
def imshow_group(X,n_per_row=10):
    # shows a group of images
    n_sample=len(X)
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,HEIGHT_PER_ROW*j))
    for i,img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
        plt.imshow(img,cmap='gray')
        plt.axis('off')
    plt.show()
X_sample_a=[img_bb(path) for path in paths_train_a[:100]]
imshow_group(X_sample_a)
X_sample_b=np.array([img_bb(path) for path in paths_train_b[:100]])
imshow_group(X_sample_b)
X_sample_e=np.array([img_bb(path) for path in paths_train_e[:100]])
imshow_group(X_sample_e)