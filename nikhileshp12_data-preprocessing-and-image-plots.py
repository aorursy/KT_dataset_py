# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

from PIL import Image

from pathlib import Path



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train_relationships.csv')

train_images =  Path('../input/train/')

test_images = Path('.../input/test/')

train_df.head()

train_df.shape
missing_folders = []

temp_list = list(train_df.p1)+list(train_df.p2)

all_list = list(set(temp_list))

all_list.sort()

for i in all_list:

    if(os.path.exists(train_images/i)==False):

        #If no such directory exists

        missing_folders.append(i)

    

print(len(missing_folders))
train_df = train_df[~train_df["p1"].isin(missing_folders)]

train_df = train_df[~train_df["p2"].isin(missing_folders)]
train_df.shape #Ignoring missing directories
temp_list = list(train_df.p1) + list(train_df.p2)

all_folders = list(set(temp_list))

all_folders.sort()

max_images = 0

for i in all_folders:

    img_list = os.listdir(train_images / i)

    if(len(img_list)>max_images):

        max_images = len(img_list)

        

print(max_images)

def show_img(n):

    try:

        n = int(n)

        img_list = os.listdir(train_images / all_folders[n])

        num_images = len(img_list)

        print(num_images)

        if(num_images<=10):

            fig,ax = plt.subplots(2,5, figsize=(20,10))

            print(ax.shape)

            for i in range(len(img_list)):

                with open(train_images / all_folders[n] / img_list[i] ,'rb') as f:

                    img = Image.open(f)

                    ax[i//5][i%5].imshow(img)

            fig.show()



        else:

            fig,ax = plt.subplots((-(-num_images//10)),10, figsize=(40,10))

            print(ax.shape)

            for i in range(len(img_list)):

                with open(train_images / all_folders[n] / img_list[i] ,'rb') as f:

                    img = Image.open(f)

                    ax[i//10][i%10].imshow(img)

            fig.show()

    except ValueError:

        n = all_folders.index(n)

        img_list = os.listdir(train_images / all_folders[n])

        num_images = len(img_list)

        print(num_images)

        if(num_images<=10):

            fig,ax = plt.subplots(2,5, figsize=(20,10))

            print(ax.shape)

            for i in range(len(img_list)):

                with open(train_images / all_folders[n] / img_list[i] ,'rb') as f:

                    img = Image.open(f)

                    ax[i//5][i%5].imshow(img)

            fig.show()



        else:

            fig,ax = plt.subplots((-(-num_images//10)),10, figsize=(40,10))

            print(ax.shape)

            for i in range(len(img_list)):

                with open(train_images / all_folders[n] / img_list[i] ,'rb') as f:

                    img = Image.open(f)

                    ax[i//10][i%10].imshow(img)

            fig.show()

        

show_img('F0002/MID1')

#show_img(0)
