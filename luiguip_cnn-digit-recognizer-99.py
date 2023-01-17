# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import math

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split



from tensorflow.keras import Model

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
BATCH_SIZE = 128

EPOCHS = 256
labeled_images = pd.read_csv('../input/digit-recognizer/train.csv')

labeled_images.info()
images = labeled_images.iloc[:, 1:]

labels = labeled_images.iloc[:, :1]

print(images.shape)

print(labels.shape)
side = int(math.sqrt(images.shape[1]))

side
images = images.values.reshape((images.shape[0], side, side, 1))
train_images, val_images, train_labels,val_labels = train_test_split(

images, labels, train_size=0.8, random_state=0) 
train_images[0].shape

train_images[0,:,:,0].shape
def show_images(i):

    img=train_images[i,:,:,0]

    plt.imshow(img, cmap='gray')

    plt.title(train_labels.iloc[i,0])



show_images(0)
train_image_generator = ImageDataGenerator(rescale=1./255,

                    rotation_range=45,

                    width_shift_range=.15,

                    height_shift_range=.15,

                    horizontal_flip=False,

                    zoom_range=0.5)

# Generator for our training data

val_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_data_gen = train_image_generator.fit(train_images)

dev_data_gen = val_image_generator.fit(val_images)
def create_model():

    

    model = Sequential()



    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                     activation ='selu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                     activation ='selu'))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))





    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='selu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='selu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))





    model.add(Flatten())

    model.add(Dense(256, activation = "selu"))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation = "softmax"))



    model.compile(loss='sparse_categorical_crossentropy',

                  metrics=['sparse_categorical_accuracy'],

                  optimizer='adam')

    return model
model = create_model()
history = model.fit_generator(

    train_image_generator.flow(train_images,train_labels, batch_size=BATCH_SIZE),

    steps_per_epoch=train_images.shape[0] // BATCH_SIZE,

    epochs=EPOCHS,

    validation_data=val_image_generator.flow(val_images, val_labels, batch_size=BATCH_SIZE),

    validation_steps=val_images.shape[0] // BATCH_SIZE

)
acc = history.history['sparse_categorical_accuracy']

val_acc = history.history['val_sparse_categorical_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(EPOCHS)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')

plt.plot(epochs_range, val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.title('Training and Validation Accuracy')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
test = pd.read_csv('../input/digit-recognizer/test.csv')

test.head()
test = test / 255.0
test_images = test.values.reshape((test.shape[0], side, side, 1))
predict = model.predict(test_images)

predict
df = pd.DataFrame(predict)

df.idxmax(1)
out_df = pd.DataFrame(data={'ImageId':df.index, 'Label':df.idxmax(1)})

out_df = out_df.set_index('ImageId')

out_df.index+=1

out_df
out_df.to_csv('results.csv', header=True)