import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import cv2

from skimage.io import MultiImage 

from PIL import Image

import openslide



import os

from tqdm.notebook import tqdm
"""

first def crop takes path to image and returns points to crop white background. 

Points are relative to shape. i.e their values are between [0,1]

"""

def crop(path):

    result = []

    imgs = openslide.OpenSlide(path)

    img = np.asarray(imgs.read_region((0,0), imgs.level_count-1, imgs.level_dimensions[-1]))

    mask = img[:,:].sum(axis=2)

    mask = (mask<(mask.max()-10)).astype('uint8')

    rect = cv2.boundingRect(mask)

    x = rect[1]/img.shape[0]

    x1 = (rect[1] + rect[3])/img.shape[0]

    y = rect[0]/img.shape[1]

    y1 = (rect[0] + rect[2])/img.shape[1]



    return x, x1, y, y1

'''

This part of code cretes dataframe of points.

You can find dataframe in input, so there is no need to execute this part of code





train_dir = '/kaggle/input/prostate-cancer-grade-assessment/train_images/'

pathes = os.listdir(train_dir)

x = []

x1 = []

y = []

y1 = []

for i in tqdm(pathes):

    points = crop(train_dir + i)

    x.append(points[0])

    x1.append(points[1])

    y.append(points[2])

    y1.append(points[3])

df = pd.DataFrame({

    'id': pathes,

    'x': x,

    'x1': x1,

    'y': y,

    'y1': y1,

})

df.to_csv('crop_points.csv')

''';
df = pd.read_csv('/kaggle/input/prostate-cancer-grade/crop_points.csv')

df
'''

Function takes relative points and shape of the neede tiff frame and returns its points 

'''

def new_points(x,x1,y,y1, shape, pad = 15):

    new_x = np.clip(int(x*shape[0]) - pad, 0, shape[0])

    new_x1 = np.clip(int(x1*shape[0]) + pad, 0, shape[0])

    new_y = np.clip(int(y*shape[1]) - pad, 0, shape[1])

    new_y1= np.clip(int(y1*shape[1]) + pad, 0, shape[1])

    return new_x, new_x1,new_y,new_y1
image_id = df.id[39] # getting id of image

imgs = openslide.OpenSlide(f'/kaggle/input/prostate-cancer-grade-assessment/train_images/{image_id}') # open image with openslide

frame = imgs.level_count - 2 # get needed level. Here you can choose which frame to use

img = np.asarray(imgs.read_region((0,0), frame, imgs.level_dimensions[frame])) # get image from multi image

x,x1,y,y1 = df.loc[df.id == image_id].values[0,1:] # get resize points

new_x, new_x1, new_y, new_y1 = new_points(x,x1,y,y1,img.shape) #scale points

new_img = img[new_x:new_x1,new_y:new_y1] # get cropped image
plt.figure(figsize = (30,20) )

plt.subplot(1,2,1)

plt.imshow(new_img)

plt.title('cropped', fontsize=20)

plt.subplot(1,2,2)

plt.imshow(img)

plt.title('original', fontsize=20);