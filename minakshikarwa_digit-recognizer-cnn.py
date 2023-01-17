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

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K



from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split
train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')

train.shape, test.shape
x_train = train.drop(['label'],1)

y_train = train['label']
train_arr = np.asarray(x_train)

test_arr = np.asarray(test)
train1 = train_arr.reshape(42000,28,28)

test1 = test_arr.reshape(28000,28,28)

train1[0]
print(y_train)

print(y_train.value_counts())

print(sns.countplot(y_train))
train1.shape, y_train.shape, test1.shape
# Plot examples of the data.

plt.figure(1, figsize=(14,3))

for i in range(10):

    plt.subplot(1,10,i+1)

    plt.imshow(train1[i], cmap='gray', interpolation='nearest')
K.image_data_format()
img_row = train1.shape[1]

img_col = train1.shape[2]

if K.image_data_format() == 'channels_first':

    train1 = train1.reshape(train1.shape[0],1,img_row,img_col)

    test1 = test1.reshape(test1.shape[0],1,img_row,img_col)

    input_shape = (1,img_row, img_col)

else:

    train1 = train1.reshape(train1.shape[0],img_row,img_col,1)

    test1 = test1.reshape(test1.shape[0],img_row,img_col,1)

    input_shape = (img_row, img_col,1)
train1.shape, test1.shape, y_train.shape
##standarisation

train1 = train1.astype('float32')

test1 = test1.astype('float32')

train1 /=255

test1 /= 255

print('train shape:', train1.shape)

print(train1.shape[0], 'train samples')

print(test1.shape[0], 'test samples')

num_classes = 10

# convert class vector to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_classes)
train1[0]
train_x, val_x, train_y, val_y = train_test_split(train1, y_train, test_size = 0.20)

train_x.shape, val_x.shape, train_y.shape, val_y.shape
model = Sequential()

model.add(Conv2D(32, kernel_size=(5,5),

                 activation = 'relu',

                 input_shape = input_shape))



model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation ='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer =keras.optimizers.Adadelta(),

              metrics= ['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=3)

filepath="/kaggle/working/bestmodel.h5"

md = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
model.fit(train_x, train_y,

          batch_size =128,

          epochs = 20,

          verbose =1,

          validation_data =(val_x,val_y),

          callbacks = [es,md]

          )
from keras import models
model1 = models.load_model("/kaggle/working/bestmodel.h5")
model1.summary()
pred = model1.predict(val_x)

pred
pred_class = model1.predict_classes(val_x)

pred_class
pd.crosstab(np.argmax(val_y, axis =1), pred_class)
y_test = model1.predict_classes(test1)

y_test
np.savetxt('pred_cnn.csv', y_test, header="Label")