# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train_ship_segmentations_v2.csv')
train_data.head()
train_samples = train_data.loc[train_data['EncodedPixels'].notnull(),:].sample(10)
train_samples
train_images = os.listdir("../input/train_v2")
len(train_images)
n_images = 10
cont = 0
fig, axs = plt.subplots(n_images,2,figsize=(15,15/2*n_images))

train_samples = train_data.loc[train_data['EncodedPixels'].notnull(),:].sample(n_images)
for ind, row_sample in train_samples.iterrows():
    #Load Image
    im = Image.open("../input/train_v2/{}".format(row_sample['ImageId']))
    px = im.load()
    print(np.asarray(im).shape)
    #Show image
    ax = axs[cont,0]
    ax.imshow(np.asarray(im))
    ax.axis('off')
    
    #Load Pixels
    pixels_array = row_sample['EncodedPixels'].split(' ')
    
    #Edit Pixels of image
    for i in np.arange(0,len(pixels_array),2):
        for pixel in range(int(pixels_array[i]), int(pixels_array[i])+int(pixels_array[i+1])):
            numpy_row = np.floor(int(pixel)/np.asarray(im).shape[0])
            numpy_column = int(pixel)-np.asarray(im).shape[0]*numpy_row
            px[numpy_row,numpy_column] = (250,0,0) 
    
    #Show image
    ax = axs[cont,1]
    ax.imshow(np.asarray(im))
    ax.axis('off')
    
    cont += 1
    
plt.show()
