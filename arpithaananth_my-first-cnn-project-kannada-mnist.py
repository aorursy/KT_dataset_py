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
import tensorflow as tf

import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Flatten, Dense, Activation, Dropout

from keras.layers import LeakyReLU

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.preprocessing import image

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.metrics import accuracy_score

from tensorflow.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping
train= pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

train.head()
test= pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

test.head()
Val= pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

Val.head()
X_train= train.drop(['label'],axis=1).values

X_train.shape
Y_train=train['label'].values

Y_train.shape
X_train = X_train.reshape(X_train.shape[0],28,28,1)

X_train.shape
np.unique(Y_train)
X_val= Val.drop(['label'],axis=1).values

X_val= X_val.reshape(X_val.shape[0],28,28,1)

X_val.shape
Y_val= Val['label'].values
X_train,X_val= X_train/255.0, X_val/255.0
X_train1, X_valid1, Y_train1, Y_valid1 = train_test_split(X_train, Y_train, test_size = 0.10, random_state=42)
train_d = ImageDataGenerator(rescale = 1./255.,

                                   rotation_range = 10,

                                   width_shift_range = 0.25,

                                   height_shift_range = 0.25,

                                   shear_range = 0.1,

                                   zoom_range = 0.25,

                                   horizontal_flip = False)

valid_d = ImageDataGenerator(rescale=1./255) 
def lr_decay(epoch):

    return initial_learningrate * 0.99 ** epoch
es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)
initial_learningrate=2e-3

batch_size = 1000

epochs = 10

input_shape = (28, 28, 1)
model= Sequential()

model.add(Conv2D(128,(3,3),input_shape=(28,28,1),activation='relu',padding='same'))

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(MaxPooling2D(pool_size=(2,2),padding='same')) 



model.add(Conv2D(256,(3,3),padding='same',activation='relu')) 

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))#2nd Convol Layer

model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.3))



model.add(Conv2D(128,(3,3),padding='same',activation='relu')) 

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))#2nd Convol Layer

model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.3))



model.add(Conv2D(64,(3,3),padding='same',activation='relu')) 

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))#2nd Convol Layer

model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.3))



model.add(Conv2D(64,(3,3),padding='same',activation='relu')) 

model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

model.add(LeakyReLU(alpha=0.1))#2nd Convol Layer

model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Dropout(0.3))



model.add(Flatten())  #Flatten #No Neurons here

model.add(Dense(64,activation='relu')) 

model.add(Dense(10,activation='softmax'))

# Final 10 Output Layer



model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit_generator(

      train_d.flow(X_train,Y_train, batch_size=batch_size),

      steps_per_epoch=100,

      epochs=epochs,

      callbacks=[LearningRateScheduler(lr_decay),es           

               ],

      validation_data=valid_d.flow(X_valid1,Y_valid1),

      validation_steps=10,  

      verbose=2)
preds_Val=model.predict_classes(X_val/255)

accuracy_score(preds_Val, Y_val)
X_test= test.drop(['id'],axis=1).values

X_test = X_test.reshape(X_test.shape[0],28,28,1)

X_test.shape
predictions = model.predict_classes(X_test/255.)
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.head()
submission.to_csv("submission.csv",index=False)