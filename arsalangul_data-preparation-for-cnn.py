# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





# Any results you write to the current directory are saved as output.
import os

import cv2

import matplotlib.pyplot as plt

%matplotlib inline
def get_images(directory):

        normal_images=[]

        pneumonia_images=[]

        for f in os.listdir(directory):

            if f == 'NORMAL':

                for image in os.listdir(directory+r'/'+f):

                    images=cv2.imread(directory+r'/'+f+r'/'+image)

                    try:

                        images=cv2.resize(images,(500,500))

                        normal_images.append(images)

                    except:

                        continue



            elif f == 'PNEUMONIA':

                for image in os.listdir(directory+r'/'+f):

                    images=cv2.imread(directory+r'/'+f+r'/'+image)

                    try:

                        images=cv2.resize(images,(500,500))

                        pneumonia_images.append(images)

                    except:

                        continue

            else:

                continue



        return np.array(normal_images),np.array(pneumonia_images)

        
n_train_images,p_train_images=get_images('../input/chest_xray/chest_xray/train')

n_test_images,p_test_images=get_images('../input/chest_xray/chest_xray/test')
print(n_train_images.shape,p_train_images.shape)
n_train_images=n_train_images.reshape((1341,-1))

p_train_images=p_train_images.reshape((3875,-1))
### Data Augmentation

train_images_list= [i for i in n_train_images]

def data_augmentation(image):

    

    image_1= image + 40

    image_2= image * 2

    

    train_images_list.append(image_1)

    train_images_list.append(image_2)

    

for i in range(n_train_images.shape[0]):

    data_augmentation(n_train_images[i])
train_images=np.array(train_images_list)

train_images = train_images[:3875]
train_images.shape
X_train= np.vstack((train_images,p_train_images))
y=np.zeros((3875,1))

y_1=np.ones((3875,1))

y=np.append(y,y_1)

y=y.reshape((7750,1))
n_test_images=n_test_images.reshape((-1,750000))

p_test_images=p_test_images.reshape((-1,750000))
X_test= np.vstack((n_test_images,p_test_images))