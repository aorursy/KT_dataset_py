# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import keras

from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import Dense 

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from sklearn.model_selection import train_test_split

train_data = pd.read_csv(r'../input/digit-recognizer/train.csv')

test_data = pd.read_csv(r'../input/digit-recognizer/test.csv')
train_data.head()
train_data.info()
y_train = train_data['label']
X_train = train_data.drop('label',axis = 1)
sns.countplot(y_train)

y_train.value_counts();
X_train.isnull().sum()
#Normalization of data:

X_train = X_train/255.0

test_data = test_data/255.0
X_train = X_train.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical

from keras.layers import Dropout
y_train = to_categorical(y_train)

num_classes = y_train.shape[1]

num_classes

X_train,X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1 ,random_state = 50)
plt.imshow(X_train[0][:,:,0])
y_train.shape
from keras.optimizers import RMSprop,Adam

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
#Applying CNN:



classifier = Sequential()
#Adding Convolution neural network to train model 

#Right now I am using two layer to train model



classifier.add(Convolution2D(filters = 32,kernel_size = (5,5), input_shape = (28,28,1), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Dropout(0.25))



classifier.add(Convolution2D(filters =64, kernel_size = (3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

classifier.add(Dropout(0.25))



# to make image flatten 

classifier.add(Flatten())



#applying fully conncected 



classifier.add(Dense(128,activation = 'relu'))

classifier.add(Dense(64,activation = 'relu'))

classifier.add(Dense(10, activation = 'softmax'))



#Compiling Conv

classifier.compile(optimizer= optimizer , loss = 'categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
X_test.shape
datagen = ImageDataGenerator(

    featurewise_center=False,

    samplewise_center= False,

    featurewise_std_normalization=False,

    samplewise_std_normalization=False,

    rotation_range=10,

    zoom_range = 0.1,

    width_shift_range=0.5,

    height_shift_range=0.5,

    horizontal_flip=False,

    vertical_flip = False,

    shear_range=0.2

    )

datagen.fit(X_train)
history = classifier.fit_generator(datagen.flow(X_train,y_train, batch_size=256),

                              epochs = 20, validation_data = (X_test,y_test))
predictions = classifier.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)