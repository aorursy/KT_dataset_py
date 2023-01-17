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
import glob

import numpy as np

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
train = []

files = glob.glob("../input/train/*.jpg") # your image path

for myFile in files:

    train_img = load_img(myFile,target_size=(128,128),color_mode="rgb")

    train_img= img_to_array(train_img)

    train_img = train_img / 255

    train.append(train_img)



train_tar = []

for myFile in files:

    train_img = load_img(myFile,target_size=(128,128),color_mode="grayscale")

    train_img= img_to_array(train_img)

    train_img = train_img / 255

    train_tar.append(train_img)

    

test = []

fikes = glob.glob("../input/valid/*.jpg")

for myfile in fikes:

    test_img = load_img(myfile,target_size=(128,128),color_mode="rgb")

    test_img= img_to_array(test_img)

    test_img = test_img / 255

    test.append(test_img)



test_tar = []

for myfile in fikes:

    test_img = load_img(myfile,target_size=(128,128),color_mode="grayscale")

    test_img= img_to_array(test_img)

    test_img = test_img / 255

    test_tar.append(test_img)

    

print("Length of Train is:",len(train))

print("Length of Test is:",len(test))
import matplotlib.pyplot as plt



plt.imshow(array_to_img(test[67]))
train = np.asarray(train)

print("The Shape of Train array is",train.shape)



train_tar = np.asarray(train_tar)

print("The Shape of Train grayscale array is",train_tar.shape)



test = np.asarray(test)

print("The Shape of Test array is",test.shape)
from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.layers import Conv2D, Flatten

from tensorflow.keras.layers import Reshape, Conv2DTranspose, Conv2D, Flatten, Reshape, MaxPooling2D, UpSampling2D

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.utils import plot_model

from tensorflow.keras import backend as K

import keras

import cv2

import numpy as np

import matplotlib.pyplot as plt

import os
# Creating Encoder Model

img_rows = 128

img_cols = 128



input_img = Input(shape = (img_rows,img_cols,3), name='encoder_input')



#Encoder 



conv3 = Conv2D(128, (2, 2), activation='relu', padding='same')(input_img)

encoded = MaxPooling2D(pool_size=(2, 2))(conv3)





shape = K.int_shape(encoded)



#Decoder

up6 = UpSampling2D((2, 2))(encoded)

decoded = Conv2D(1, (2, 2), activation='tanh', padding='same')(up6)





# instantiate encoder model

encoder = Model(input_img, decoded, name='encoder')

encoder.summary()

import keras

from keras import optimizers



#Fitting the model

encoder.compile(optimizer='adam', loss='mse',metrics=['mae'])
hist = encoder.fit(train,train_tar,verbose = 1, epochs = 200,batch_size = 32, validation_split = 0.40)
# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for accuracy

plt.plot(hist.history['mae'])

plt.plot(hist.history['val_mae'])

plt.title('model Mean Absolute Error')

plt.ylabel('mae')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
decoded = encoder.predict(test,verbose=0)
plt.imshow(array_to_img(test[3]))
plt.imshow(array_to_img(decoded[3]))
plt.imshow(array_to_img(test_tar[3]))
%matplotlib inline



#Plotting some of the images from the test dataset and saving the output of the images

for i in range(1,9):

    fig = plt.figure(figsize=(10,10))

    

    ax1 = fig.add_subplot(3,3,1)

    ax1.set_title("RGB Image")

    ax1.imshow(array_to_img(test[i]))

    

    ax2 = fig.add_subplot(3,3,2)

    ax2.set_title("Ground_Truth")

    ax2.imshow(array_to_img(test_tar[i]))

    

    ax3 = fig.add_subplot(3,3,3)

    ax3.set_title("Converted GrayScale")

    ax3.imshow(array_to_img(decoded[i]))

    

    fig.savefig('reconstructed_img'+ str(i)+".png")