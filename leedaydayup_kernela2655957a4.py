# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from keras.preprocessing.image import ImageDataGenerator

from keras.applications.densenet import DenseNet121



import os

from tqdm import tqdm,tqdm_notebook

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv")

train_df.head()



x_train,x_test,y_train,y_test = train_test_split(

    train_df['id'],

    train_df['has_cactus'],

    test_size = 0.2,

    random_state = 1)
X_train = []

for images in tqdm(x_train):

    img = plt.imread('../input/train/train/' + images)

    X_train.append(img)

    



X_test = []

for images in tqdm(x_test):

    img = plt.imread('../input/train/train/' + images)

    X_test.append(img)

    

X_train = np.array(X_train)

X_test = np.array(X_test)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')



X_train = X_train/255

X_test = X_test/255
augmentations = ImageDataGenerator(

    vertical_flip=True,

    horizontal_flip=True,

    zoom_range=0.1)



augmentations.fit(X_train)