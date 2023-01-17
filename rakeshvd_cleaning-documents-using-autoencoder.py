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
import os

import glob

import cv2

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from keras.preprocessing import image

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import MaxPooling2D, Dropout, UpSampling2D
training_images = glob.glob('/kaggle/input/denoising-dirty-documents/train/*.png')

cleaned_images = glob.glob('/kaggle/input/denoising-dirty-documents/train_cleaned/*.png')

test_images = glob.glob('/kaggle/input/denoising-dirty-documents/test/*.png')
def load_image(path):

    image_list = np.zeros((len(path), 258, 540, 1))

    for i, fig in enumerate(path):

        img = image.load_img(fig, color_mode='grayscale', target_size=(258, 540))

        x = image.img_to_array(img).astype('float32')

        x = x / 255.0

        image_list[i] = x

    

    return image_list



x_train = load_image(training_images)

y_train = load_image(cleaned_images)

x_test = load_image(test_images)



print(x_train.shape, x_test.shape)

def train_val_split(x_train, y_train):

    rnd = np.random.RandomState(seed=42)

    perm = rnd.permutation(len(x_train))

    train_idx = perm[:int(0.8 * len(x_train))]

    val_idx = perm[int(0.8 * len(x_train)):]

    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]



x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)

print(x_train.shape, x_val.shape)
img_rows = 258

img_cols = 540

channels = 1

img_shape = (img_rows, img_cols, channels)
input_layer = Input(shape=img_shape)

        

# encoder

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)

x = MaxPooling2D((2, 2), padding='same')(x)



# decoder

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)





output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        



autoencoder_model =Model(input_layer, output_layer)

autoencoder_model.compile(loss='mse', optimizer=Adam(lr=0.001))

history = autoencoder_model.fit(x_train, y_train,

                                             batch_size=32,

                                             epochs=250,

                                             validation_data=(x_val, y_val),

                                             callbacks=[EarlyStopping(monitor='val_loss',

                                       min_delta=0,

                                       patience=20,

                                       verbose=1, 

                                       mode='auto')])

autoencoder_model.summary()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



preds = autoencoder_model.predict(x_test)
preds = autoencoder_model.predict(x_test)

print(preds)
x_test_0 = x_test[13] * 255.0

x_test_0 = x_test_0.reshape(258, 540)

x_test_1 = x_test[55] * 255.0

x_test_1 = x_test_1.reshape(258, 540)

x_test_2 = x_test[34] * 255.0

x_test_2 = x_test_2.reshape(258, 540)

x_test_3 = x_test[40] * 255.0

x_test_3 = x_test_3.reshape(258, 540)

fig = plt.figure(figsize=(15, 15))

fig.add_subplot(2,2,1)

plt.imshow(x_test_0,cmap='gray')

ax2 = fig.add_subplot(2,2,2)

ax2.imshow(x_test_1,cmap='gray')

ax3 = fig.add_subplot(2,2,3)

ax3.imshow(x_test_2,cmap='gray')

ax4 = fig.add_subplot(2,2,4)

ax4.imshow(x_test_3,cmap='gray')
preds_0 = preds[13] * 255.0

preds_0 = preds_0.reshape(258, 540)

preds_1 = preds[55] * 255.0

preds_1 = preds_1.reshape(258, 540)

preds_2 = preds[34] * 255.0

preds_2 = preds_2.reshape(258, 540)

preds_3 = preds[40] * 255.0

preds_3 = preds_3.reshape(258, 540)

fig = plt.figure(figsize=(15, 15))

fig.add_subplot(2,2,1)

plt.imshow(preds_0,cmap='gray')

ax2 = fig.add_subplot(2,2,2)

ax2.imshow(preds_1,cmap='gray')

ax3 = fig.add_subplot(2,2,3)

ax3.imshow(preds_2,cmap='gray')

ax4 = fig.add_subplot(2,2,4)

ax4.imshow(preds_3,cmap='gray')