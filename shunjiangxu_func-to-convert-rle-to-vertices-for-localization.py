# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 # image processing
from skimage.io import imread
import matplotlib.pyplot as plt
import operator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def rle_decode(mask_rle, mask_value=255, shape=(768,768)):
    ## this function convert RLE encoding into image_mask
    if type(mask_rle) == float: return None  ## My way of reading 
    
    if type(mask_rle) == str:
        mask_rle = [mask_rle]
    
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)  ## initialzing images to all 0
    for mask in mask_rle:
        s = mask.split()
        starts, lengths = np.asarray(s[0::2], dtype=int)-1, np.asarray(s[1::2], dtype=int)
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = mask_value
    return img.reshape(shape).T

def rle_to_vertices(mask_rle, return_img=False, shape=(768,768)):
    ## This function finding out the center, length, width, angle of the RLE and return these parameters plus the image with box countour
    if type(mask_rle) == float: return None
    mask_img = rle_decode(mask_rle, shape=shape) # Generate masked images
    ret, mask_img = cv2.threshold(mask_img, 127, 255, 0) 
    im2, contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        ## Storing center information so to assign ship to proper grid
        #center_x = rect[0][1]
        #len_x = rect[0][0]
        #center_y = rect[1][1]
        #len_y = rect[1][0]
        center_x = rect[0][0]
        center_y = rect[0][1]
        len_x = rect[1][0]
        len_y = rect[1][1]
        angle = rect[2]
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        boxes.append([center_x, center_y, len_x, len_y, angle]) 
    if return_img:
        for center_x, center_y, len_x, len_y, angle in boxes:
            #box = cv2.boxPoints(((len_x, center_x), (len_y, center_y),angle))
            box = cv2.boxPoints(((center_x, center_y), (len_x, len_y),angle))
            box = np.int0(box)
            cv2.drawContours(mask_img,[box],0,200,1)
            #mask_img[int(center_x)-5:int(center_x)+6, int(center_y)-5:int(center_y)+6] = 200
        return boxes, mask_img
    else:
        return boxes

def draw_mask(boxes, shape=(768,768)):
    mask_img = np.zeros(shape, dtype=np.uint8)
    if len(boxes) == 0:
        return mask_img
    for center_x, center_y, len_x, len_y, angle in boxes:
        box = cv2.boxPoints(((center_x, center_y), (len_x, len_y),angle))
        box = np.int0(box)
        cv2.drawContours(mask_img,[box],0,200,1)
        #mask_img[int(center_x)-5:int(center_x)+6, int(center_y)-5:int(center_y)+6] = 200
    return mask_img
masks = pd.read_csv('../input/train_ship_segmentations_v2.csv')
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])
masks['path'] = masks['ImageId'].map(lambda x: os.path.join('../input/train_v2', x))
masks.head()
## Show image with smallest & largest size ships alongside with masks.
## The purpose is the test out the rle_to_vertices function.

ship_size_list = []
i = 0

## Finding out the size of the ships and put that into a list
for pixels in masks['EncodedPixels']:
    if type(pixels) == str:
        ship_size_list.append([i, sum([int(length) for length in pixels.split()[1::2]])])
    i += 1
print('Number of ships: {}'.format(len(ship_size_list)))
min_index, min_value = min(ship_size_list, key=operator.itemgetter(1))
print('Min size of ship: {}'.format(min_value))
print(masks.iloc[min_index])
max_index, max_value = max(ship_size_list, key=operator.itemgetter(1))
print('Max size of ship: {}'.format(max_value))
print(masks.iloc[max_index])

min_pixel_list = list(masks[masks['path'] == masks.loc[min_index, 'path']]['EncodedPixels'])
min_mask = rle_decode(min_pixel_list)
min_image = imread(masks.loc[min_index, 'path'])
print('list length is {}'.format(len(min_pixel_list)))
print(min_pixel_list)
plt.figure(figsize=(50,50))
plt.subplot(2,2,1)
#plt.imshow(min_mask[530:550,35:55])
plt.imshow(min_mask)
plt.subplot(2,2,2)
#plt.imshow(min_image[530:550,35:55,:])
plt.imshow(min_image)

max_pixel_list = list(masks[masks['path'] == masks.loc[max_index, 'path']]['EncodedPixels'])
max_mask = rle_decode(max_pixel_list)
max_image = imread(masks.loc[max_index, 'path'])
print('list length is {}'.format(len(max_pixel_list)))
#print(min_pixel_list)
plt.figure(figsize=(50,50))
plt.subplot(2,2,3)
#plt.imshow(min_mask[530:550,35:55])
plt.imshow(max_mask)
plt.subplot(2,2,4)
#plt.imshow(min_image[530:550,35:55,:])
plt.imshow(max_image)
# Testing functions, look at the green countour of the mask. The vertices parameter can be used in training a localization model.
vertices, img = rle_to_vertices(max_pixel_list, return_img=True)
plt.figure(figsize=(50,50))
plt.imshow(img)

