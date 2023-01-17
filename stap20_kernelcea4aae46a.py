# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import cv2

from pathlib import Path

import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Define path to the data directory

data_dir = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray')



# Path to train directory (Fancy pathlib...no more os.path!!)

train_dir = data_dir / 'train'



# Path to validation directory

val_dir = data_dir / 'val'



# Path to test directory

test_dir = data_dir / 'test'
# Get the path to the normal and pneumonia sub-directories

normal_cases_dir = train_dir / 'NORMAL'

pneumonia_cases_dir = train_dir / 'PNEUMONIA'



# Get the list of all the images

normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')



# An empty list. We will insert the data into this list in (img_path, label) format

train_data = []



# Go through all the normal cases. The label for these cases will be 0

for img in normal_cases:

    train_data.append((img,0))



# Go through all the pneumonia cases. The label for these cases will be 1

for img in pneumonia_cases:

    train_data.append((img, 1))



# Get a pandas dataframe from the data we have in our list 

train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)



# Shuffle the data 

train_data = train_data.sample(frac=1.).reset_index(drop=True)



# How the dataframe looks like?

train_data.head()
# Get the path to the normal and pneumonia sub-directories

normal_cases_dir = train_dir / 'NORMAL'

pneumonia_cases_dir = train_dir / 'PNEUMONIA'



# Get the list of all the images

normal_cases = normal_cases_dir.glob('*.jpeg')

pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')





pneumonia_imgs = []

for img in pneumonia_cases:

    im = cv2.imread(str(img),0)

    im = cv2.resize(im,(100,100))

    pneumonia_imgs.append(im)

    

normal_imgs = []

for img in normal_cases:

    im = cv2.imread(str(img),0)

    im = cv2.resize(im,(100,100))

    normal_imgs.append(im)

    

print(len(pneumonia_imgs))

print(len(normal_imgs))



X_train= np.array(pneumonia_imgs+normal_imgs).reshape((-1,100,100,1))

y_train=np.concatenate((np.ones(len(pneumonia_imgs)),np.zeros(len(normal_imgs)))).reshape((-1,1))



print(X_train.shape)

print(y_train.shape)

def unison_shuffled_copies(a, b):

    assert len(a) == len(b)

    p = np.random.permutation(len(a))

    return a[p], b[p]





X_train,y_train=unison_shuffled_copies(X_train,y_train)



X_test=X_train[-1000:]

y_test=y_train[-1000:]



X_train=X_train[:-1000]

y_train=y_train[:-1000]



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)


from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam



model = Sequential()

model.add(Conv2D(32,4, input_shape=(100, 100,1)))

model.add(  MaxPooling2D(pool_size=2))

model.add(Conv2D(32,4, input_shape=(100, 100,1)))

model.add(  MaxPooling2D(pool_size=2))



model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=1, batch_size=256, validation_data=(X_test,y_test))