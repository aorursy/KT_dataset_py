from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.patches as patches



%matplotlib inline 
!ls ../input/
!ls ../input/CUB_200_2011/
df = pd.read_csv('../input/image_data.csv')

df.head()
rows = 5

cols = 10

size = 2

f = plt.figure(figsize=(cols*size,rows*size))

img_paths = df['path'].values

# np.random.shuffle(img_paths)

path = "../input/CUB_200_2011/images/"

for i, img_path in enumerate(img_paths):

    img = cv2.cvtColor(cv2.imread(path+img_path), cv2.COLOR_BGR2RGB)

    

    ax = f.add_subplot(rows, cols, i+1)

    data = df[df['path'] == img_path]

    x, y = data['xmin'].values, data['ymin'].values

    w, h = data['width'].values, data['height'].values

    

    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')

    ax.add_patch(rect)

    

    plt.imshow(img)

    plt.axis('off')

    if i == rows*cols -1:

        break

    
rows = 5

cols = 10

size = 2

f = plt.figure(figsize=(cols*size,rows*size))

img_paths = df['path'].values

path = "../input/segmentations/"

for i, img_path in enumerate(img_paths):

    img = cv2.imread(path+img_path[:-3]+"png")

    

    ax = f.add_subplot(rows, cols, i+1)

    

    plt.imshow(img)

    plt.axis('off')

    if i == rows*cols -1:

        break