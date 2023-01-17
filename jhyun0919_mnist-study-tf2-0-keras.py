!pip install tensorflow-gpu==2.0.0-beta1
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt #for plotting

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

import tensorflow as tf
train = pd.read_csv("../input/train.csv")

print(train.shape)

train.head()
z_train = Counter(train['label'])

z_train
sns.countplot(train['label'])
#loading the dataset.......(Test)

test= pd.read_csv("../input/test.csv")

print(test.shape)

test.head()
x_train = (train.ix[:,1:].values).astype('float32') # all pixel values

y_train = train.ix[:,0].values.astype('int32') # only labels i.e targets digits



x_test = test.values.astype('float32')
# preview the images first

plt.figure(figsize=(12,10))

x, y = 10, 4

for i in range(40):  

    plt.subplot(y, x, i+1)

    plt.imshow(x_train[i].reshape((28,28)),interpolation='nearest')

plt.show()
print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)



print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')
X_train = x_train.reshape(x_train.shape[0], 28, 28,1)

X_test = x_test.reshape(x_test.shape[0], 28, 28,1)



print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=42)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau
num_of_classes = 10



epochs = 20

batch_size = 32



input_shape = (28, 28, 1)
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2)))

model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.25))

model.add(Dense(num_of_classes, activation='softmax'))



model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,

              optimizer=tf.keras.optimizers.RMSprop(),

              metrics=['accuracy'])
model.summary()
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)
datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    rotation_range=15, # randomly rotate images in the range (degrees, 0 to 180)

    zoom_range = 0.1, # Randomly zoom image 

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    horizontal_flip=False,  # randomly flip images

    vertical_flip=False)  # randomly flip images
datagen.fit(X_train)



history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, 

                              validation_data = (X_val,Y_val),

                              verbose = 1, 

                              steps_per_epoch=X_train.shape[0] // batch_size, 

                              callbacks=[learning_rate_reduction])
final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
#get the predictions for the test data

Y_test = model.predict_classes(X_test)
ids = [x for x in range(1, Y_test.shape[0] + 1)]

pd_submit = pd.DataFrame({'ImageId':ids, 'Label':Y_test})

pd_submit.to_csv("submission.csv", index=False)