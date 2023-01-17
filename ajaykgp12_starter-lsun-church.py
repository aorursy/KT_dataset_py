# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import time

import matplotlib.pyplot as plt

import random



# Any results you write to the current directory are saved as output.
%%time

X_train = np.load('../input/church_outdoor_train_lmdb_color_64.npy')

print(X_train.shape)



def display_random_images( X_train):

        plt.figure(figsize=(16,16))

        for i in range(64):

            plt.subplot(8,8,i+1)

            random_idx = random.randint(0, X_train.shape[0] -1)

            image = X_train[random_idx,:,:,:]            

            plt.axis('off')

            plt.imshow(image)

       

        plt.show()

        

display_random_images(X_train)