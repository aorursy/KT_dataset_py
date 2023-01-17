#Import some necessary Modules

import os

import cv2

import keras

import numpy as np

import pandas as pd

import random as rn

import random as rn

from PIL import Image

from tqdm import tqdm



from tensorflow.python.keras import backend as K

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



from tensorflow.python.keras.callbacks import ReduceLROnPlateau



from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.applications.resnet50 import preprocess_input

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D,GlobalAveragePooling2D, MaxPool2D,BatchNormalization,Dropout



resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

print(os.listdir("../input"))
def prep_cnn_data(df, n_x, n_c, path):

    """

    This function loads the image jpg data into tensors

    """

    # initialize tensors

    tensors = np.zeros((df.shape[0], n_x, n_x, n_c))

    # load image as arrays into tensors

    for i in range(df.shape[0]):

        pic = load_img(path+df.iloc[i]['id'])

        pic_array = img_to_array(pic)

        tensors[i,:] = pic_array

    # standardize the values by dividing by 255

    tensors = tensors / 255.

    return tensors
# prepare the train data for CNN

train_df = pd.read_csv('../input/aerial-cactus-identification/train.csv')

test_df = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

train = prep_cnn_data(train_df, 32, 3, path='../input/aerial-cactus-identification/train/train/')

train_y = train_df['has_cactus'].values

test = prep_cnn_data(test_df, 32, 3, path='../input/aerial-cactus-identification/test/test/')
# # Display first 15 images of moles, and how they are classified



fig=plt.figure(figsize=(18, 5))

columns = 15

rows = 3



for i in range(1, columns*rows +1):

    ax = fig.add_subplot(rows, columns, i)

    if train_y[i] == 1:

        ax.title.set_text('1')

    else:

        ax.title.set_text('0')

    plt.imshow((train[i]), interpolation='nearest')

plt.show()
datagen = ImageDataGenerator(

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True)  # randomly flip images

# resnet-50

#--------------------------------------------------------

# from tensorflow.python.keras import models

# from tensorflow.python.keras import layers



# model = Sequential()

# model.add(ResNet50(include_top=False,input_tensor=None,

#                    input_shape=(32,32,3),pooling='avg',

#                    classes=2,weights=resnet_weights_path))

# model.add(layers.Flatten())

# model.add(BatchNormalization())

# model.add(layers.Dense(512, activation='relu'))

# model.add(layers.Dropout(rate=0.4))

# model.add(BatchNormalization())

# model.add(layers.Dense(512, activation='relu'))

# model.add(Dropout(0.5))

# model.add(BatchNormalization())

# model.add(layers.Dense(1, activation='sigmoid'))



# model.layers[0].trainable = False



del model
# plain CNN-1

#--------------------------------------------------------

# from keras import models

# from keras import layers



# model = models.Sequential()

# model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))

# model.add(layers.Conv2D(32, kernel_size=3, padding='valid', activation='relu'))

# model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

# model.add(layers.Dropout(rate=0.25))

# model.add(layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))

# model.add(layers.Conv2D(64, kernel_size=5, padding='valid', activation='relu'))

# model.add(layers.MaxPooling2D(pool_size=(2,2), strides=2))

# model.add(layers.Dropout(rate=0.25))

# model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))

# model.add(layers.Conv2D(128, kernel_size=3, padding='valid', activation='relu'))

# model.add(layers.Flatten())

# model.add(layers.Dense(512, activation='relu'))

# model.add(layers.Dropout(rate=0.4))

# model.add(layers.Dense(512, activation='relu'))

# model.add(layers.Dense(1, activation='sigmoid'))
# plain CNN-2

#-----------------------------------------------------------------

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (32,32,3)))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))
# Compile the model

model.compile(optimizer = 'adam', loss = "binary_crossentropy", metrics=["accuracy"])
batch_size=256

epochs=200

x_train=train[3500:]

y_train=train_y[3500:]

x_val=train[:3500]

y_val=train_y[:3500]

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.7)

History = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),

                              epochs= epochs, steps_per_epoch=x_train.shape[0]//batch_size,

                              validation_data=(x_val,y_val) ,callbacks=[red_lr])
pred=model.predict(test)

test_csv = pd.read_csv('../input/aerial-cactus-identification/sample_submission.csv')

df=pd.DataFrame({'id':test_csv['id'] })

df['has_cactus']=pred

df.to_csv("submission.csv",index=False)