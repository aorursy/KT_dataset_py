import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2 

import matplotlib.pyplot as plt



np.random.seed(55)

import os



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

#data.head()
view_count =  15

i_chk = np.random.randint(0,len(data), size = view_count)

sample_imgs = []

ben_sample_imgs = []

file_list = ['../input/train_images/{}.jpg'.format(data['image_id'].values[i_chk[i]]) for i in range(view_count)]



for i in range(view_count) :

    sample_img = cv2.imread(file_list[i])

    sample_img = cv2.cvtColor(sample_img,cv2.COLOR_BGR2RGB)

    sample_imgs.append(sample_img)

    ben_sample_imgs.append(cv2.addWeighted (sample_img,4, cv2.GaussianBlur(sample_img, (0,0) , 10) ,-4 ,128))
for i in range(15) :

    fig , ax = plt.subplots(1,2,figsize = (12,15))

    ax[0].imshow(sample_imgs[i])

    ax[1].imshow(ben_sample_imgs[i])

    #plt.autoscale(tight = 'True' , axis = 'y')

    ax[0].set_title(data['image_id'].values[i_chk[i]], y = 1)

    ax[1].set_title(str(data['image_id'].values[i_chk[i]]) + ' with preprocess', y = 1)

    