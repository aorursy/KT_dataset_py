# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

img_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    if(dirname == "/kaggle/input/pokemon-mugshots-from-super-mystery-dungeon/smd"):

        for filename in filenames:

            img_files.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os,shutil,glob

from tqdm.notebook import tqdm

from skimage.transform import resize,rescale

import matplotlib.pyplot as plt

import matplotlib.image as mpimg 



import numpy as np

print(len(img_files))
images = []

for imgfile in tqdm(img_files[0:500]):

    img = mpimg.imread(imgfile)

    images.append(resize(img[:,:,0],(32,32,1)))



print(np.shape(images))
from sklearn.model_selection import train_test_split



X_train, X_test = train_test_split(

    images,

    test_size=0.2,

    shuffle=True,

    random_state=42,

)

# X_train, X_val = train_test_split(

#     X_train, 

#     test_size=0.2,

#     shuffle=True,

#     random_state=42,)



X_train = np.array(X_train)

X_test = np.array(X_test)

# X_val = np.array(X_val)





X_train = X_train.astype('float32') / 255.

X_test = X_test.astype('float32') / 255.
print(X_train.shape,X_test.shape)
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.keras.models import Model
autoencoder = None

input_img = Input(shape=(32,32,1))  # adapt this if using `channels_first` image data format



# 1 -----------------------------------------------------------------------------------------

x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPooling2D((2, 2), padding='same')(x)



# at this point the representation is (4, 4, 8) i.e. 128-dimensional



x = Conv2D(8, (4, 4), activation='relu', padding='same')(encoded)

x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)

decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)



# 2 -----------------------------------------------------------------------------------------

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

# x = MaxPooling2D((2, 2), padding='same')(x)

# # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

# # x = MaxPooling2D((2, 2), padding='same')(x)

# # x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

# # x = MaxPooling2D((2, 2), padding='same')(x)

# # x = Dense(32, activation='relu')(x)

# # x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

# # x = UpSampling2D((2, 2))(x)

# # x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

# # x = UpSampling2D((2, 2))(x)

# # x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

# x = UpSampling2D((2, 2))(x)

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)

# # x = UpSampling2D((2, 2))(x)

# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)





# 3 -----------------------------------------------------------------------------------------



# x = Dense(32, activation='relu')(input_img)

# # x = Dense(64, activation='relu')(x)

# # encoded = Dense(256, activation='relu')(x)



# # this is the loss reconstruction of the input

# # x = Dense(64, activation='relu')(encoded)

# # x = Dense(512, activation='relu')(encoded)

# decoded = Dense(1024, activation='sigmoid')(x)



# -------------------------------------------------------------------------------------------

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy',

              metrics=['accuracy'])



autoencoder.summary()
history = autoencoder.fit(X_train, X_train,

                batch_size=4,

                epochs = 100,

#                 pretrain_learning_rate = 0.01,

#                 finetune_learning_rate = 0.01,

#                 corruption_level = 0.2,

                shuffle=True,

                validation_data=(X_test, X_test))
plt.plot(history.history['accuracy'], label='loss')

plt.plot(history.history['val_accuracy'], label = 'val_loss')

plt.xlabel('Epoch')

plt.ylabel('loss')

plt.ylim([0, 0.5])

plt.legend(loc='upper right')
decoded_imgs = autoencoder.predict(X_train)



n = 10

plt.figure(figsize=(20, 4))

for i in range(1,n):

    # display original

    ax = plt.subplot(2, n, i)

    plt.imshow(X_train[i][:,:,0])

#     plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)



    # display reconstruction

    ax = plt.subplot(2, n, i + n)

    plt.imshow(decoded_imgs[i][:,:,0])

#     plt.gray()

    ax.get_xaxis().set_visible(False)

    ax.get_yaxis().set_visible(False)

plt.show()