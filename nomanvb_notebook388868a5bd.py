# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import glob

from sklearn import cluster

import cv2

import skimage.measure as sm

import multiprocessing

import random

import matplotlib.pyplot as plt

from scipy.misc import imread

import seaborn as sns

%matplotlib inline
new_style = {'grid':False}

plt.rc('axes',**new_style)



#Function to show 4 images

def show_four(imgs, title):

    #select_imgs = [np.random.choice(imgs) for _ in range(4)]

    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(4)]

    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 3))

    plt.suptitle(title, size=20)

    for i, img in enumerate(select_imgs):

        ax[i].imshow(img)

        

print ("done")
# Function to show 8 images

def show_eight(imgs, title):

    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(8)]

    _, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(20, 6))

    plt.suptitle(title, size=20)

    for i, img in enumerate(select_imgs):

        ax[i // 4, i % 4].imshow(img)

print("done")
select = 500  # load 500

train_files= sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:select]

train = np.array([imread(img) for img in train_files])

print('Length of train{}'.format(len(train)))

print('Sizes in train:')

    
shapes = np.array([str(img.shape) for img in train])

pd.Series(shapes).value_counts()

#print(pd.Series(shapes).value_counts())

print( pd.Series(shapes).unique())

for uniq in pd.Series(shapes).unique():

    show_four(train[shapes == uniq], 'Images with shape: {}'.format(uniq))

    plt.show()

    
# Function for computing distance between images

def compare(args):

    img, img2 = args

    img = (img - img.mean()) / img.std()

    img2 = (img2 - img2.mean()) / img2.std()

    return np.mean(np.abs(img - img2))

print('Done')
# Resize the images to speed it up.

train = [cv2.resize(img, (224, 224), cv2.INTER_LINEAR) for img in train]
train[1].std()
#[(train[i]) for i in range(len(train) // 2)]

#imshow(train[1])

plt.imshow(train[100])

plt.show()