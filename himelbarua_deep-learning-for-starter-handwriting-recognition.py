# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential 

from keras.layers import Dense, Lambda, Dropout, Flatten, Activation

from keras.optimizers import Adam, RMSprop

from sklearn.model_selection import train_test_split

from keras import backend as k

from keras.preprocessing.image import ImageDataGenerator 

#train data

df=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

[r_train,c_train]=df.shape

print('Number of train row is ' + str(r_train))

print('Number of train column is ' + str(c_train))
#test data

df1=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

[r_test, c_test]=df1.shape

print('Number of test row is ' + str(r_test))

print('Number of test column is ' + str(c_test))
#label isolation

X_train=(df.iloc[:,1:].values).astype('float32')# other values are float type

Y_train=(df.iloc[:,0].values).astype('int32')# only lables are integer type

X_test=(df1.iloc[:,:].values).astype('float32')

print('X_train shape: '+ str(np.shape(X_train)))

print('X_test shape:' + str(np.shape(X_test)))
#data visualization

# as the input datas are images and the each images are in 1D vector fomat to visulaize them they need to be reshaped

r_image=28

c_image=28

X_train=X_train.reshape(r_train,r_image,c_image)



for i in range (6,9):

    plt.subplot(330+(i+1))

    plt.imshow(X_train[i])

    plt.title(Y_train[i])

#data standarization

mean_px=X_train.mean().astype(np.float32)

std_px=X_train.std().astype(np.float32)



def standardize(x):

    (x-mean_px)/std_px
from keras.utils.np_utils import to_categorical



#one hot encoding 

# as this problem involves classification of images, the output layer has to have all the possible class so that the model can predict the 

#probability of each class

Y_train.shape

from keras.utils.np_utils import to_categorical

Y_train1=to_categorical(Y_train)

Y_train1.shape

#num_class=Y_train.shape[1]

#num_class
#designing neural netwrok (CNN)

from keras.layers import Convolution2D, MaxPooling2D 

from keras.utils import np_utils

model=Sequential()

model.add(Convolution2D(32,3,3, activation='relu',input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))





#compiling

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#model.fit(X_train, Y_train, 

#batch_size=32, nb_epoch=10, verbose=1)

model.fit(X_train, Y_train1, batch_size=32, epochs=10)

#score = model.evaluate(X_test, Y_test, verbose=0)

score, acc =model.evaluate(X_train,Y_train, verbose=0)

#score, acc = model.evaluate(x_test, y_test,batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)