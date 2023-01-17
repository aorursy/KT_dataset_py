import os

import numpy as np

import matplotlib.pyplot as plt

import cv2
base_dir = "/kaggle/input/image-super-resolution/dataset/"
def load_data(path):

    high_res_images = []

    low_res_images = []

    for dirname, _, filenames in os.walk(path+'low_res'):

        for filename in filenames:

            img = cv2.imread(os.path.join(dirname, filename))

            img = process_image(img)

            low_res_images.append(img)

        

    for dirname, _, filenames in os.walk(path+'high_res'):

        for filename in filenames:

            img = cv2.imread(os.path.join(dirname, filename))

            img = process_image(img)

            high_res_images.append(img)

    

    return np.array(low_res_images), np.array(high_res_images)
def process_image(image):

    return image/255
train_x, train_y =  load_data(base_dir+'train/')

val_x, val_y = load_data(base_dir+'val/')
fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('Image Comparison')

ax1.imshow(train_x[11])

ax1.title.set_text("low-res image ")

ax2.imshow(train_y[11])

ax2.title.set_text("high-res image ")
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, UpSampling2D, add

from tensorflow.keras.models import Model

from tensorflow.keras import regularizers

import tensorflow as tf



def build_model():

    input_img = Input(shape=(256, 256, 3))

    l1 = Conv2D(64, (3, 3), padding='same', activation='relu', 

                activity_regularizer=regularizers.l1(10e-10))(input_img)

    l2 = Conv2D(64, (3, 3), padding='same', activation='relu', 

                activity_regularizer=regularizers.l1(10e-10))(l1)



    l3 = MaxPooling2D(padding='same')(l2)

    l3 = Dropout(0.3)(l3)

    l4 = Conv2D(128, (3, 3),  padding='same', activation='relu', 

                activity_regularizer=regularizers.l1(10e-10))(l3)

    l5 = Conv2D(128, (3, 3), padding='same', activation='relu', 

                activity_regularizer=regularizers.l1(10e-10))(l4)



    l6 = MaxPooling2D(padding='same')(l5)

    l7 = Conv2D(256, (3, 3), padding='same', activation='relu', 

                activity_regularizer=regularizers.l1(10e-10))(l6)

    

    l8 = UpSampling2D()(l7)



    l9 = Conv2D(128, (3, 3), padding='same', activation='relu',

                activity_regularizer=regularizers.l1(10e-10))(l8)

    l10 = Conv2D(128, (3, 3), padding='same', activation='relu',

                 activity_regularizer=regularizers.l1(10e-10))(l9)



    l11 = add([l5, l10])

    l12 = UpSampling2D()(l11)

    l13 = Conv2D(64, (3, 3), padding='same', activation='relu',

                 activity_regularizer=regularizers.l1(10e-10))(l12)

    l14 = Conv2D(64, (3, 3), padding='same', activation='relu',

                 activity_regularizer=regularizers.l1(10e-10))(l13)



    l15 = add([l14, l2])



    decoded = Conv2D(3, (3, 3), padding='same', activation='relu', 

                     activity_regularizer=regularizers.l1(10e-10))(l15)





    model = Model(input_img, decoded)

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.compile(optimizer='adam', loss='mean_squared_error')

    

    return model
with tf.device('/device:GPU:0'):

    model = build_model()

    train_x , train_y = train_x , train_y

    val_x , val_y = val_x, val_y
model.summary()
def train(train_x, train_y, epochs = 1, batch_size = 32, shuffle = False):

    model.fit(train_x, train_y,

                            epochs= epochs,

                            batch_size=batch_size,

                            shuffle=shuffle)
train(train_x, train_y, epochs = 25, batch_size = 8, shuffle = True)
predict_y = model.predict(val_x)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.imshow(val_x[11])

ax1.title.set_text("low-res image ")

ax2.imshow(val_y[11])

ax2.title.set_text("high-res image ")

ax3.imshow(predict_y[11])

ax3.title.set_text("model's output")