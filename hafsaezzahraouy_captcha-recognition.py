# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
from skimage import io

io.imshow('/kaggle/input/captcha-version-2-images/samples/samples/wfy5m.png')
img=io.imread('/kaggle/input/captcha-version-2-images/samples/samples/wfy5m.png')
from skimage.filters import threshold_otsu

from skimage.color import rgb2gray

img_gray=rgb2gray(img)

io.imshow(img_gray)

thresh = threshold_otsu(img_gray)

# Apply thresholding to the image

binary = img_gray >thresh

# Show the image

io.imshow(binary)
#edge detection

from skimage.filters import sobel

img_edge=sobel(binary)

io.imshow(img_edge)
#gaussian filter

from skimage.filters import gaussian

img_gauss=gaussian(img, multichannel=True)

io.imshow(img_gauss)
# Import the module and function

from skimage.restoration import denoise_tv_chambolle



# Apply total variation filter denoising

denoised_image = denoise_tv_chambolle(img, 

                                      multichannel=True)



io.imshow(denoised_image)

io.imshow(img)
# Import the Sequential model and Dense layer

from keras import layers

from keras.models import Model

from keras.layers import BatchNormalization

img_shape=(50,200,1)

image = layers.Input(shape=img_shape)

conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(image)

mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 100x25

conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)

mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 50x13

conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)

bn = layers.BatchNormalization()(conv3)

mp3 = layers.MaxPooling2D(padding='same')(bn)

flat = layers.Flatten()(mp3)
outs = []

#we have 5 letters

for i in range(5):

    dens1 = layers.Dense(64, activation='relu')(flat)

    drop = layers.Dropout(0.5)(dens1)

    res = layers.Dense(36, activation='sigmoid')(drop)

    outs.append(res)
# Compile model and return it

model = Model(image, outs)

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
import string

symbols = string.ascii_lowercase + string.digits

num_symbols = len(symbols)

print(num_symbols)
import cv2

n_samples = len(os.listdir('../input/captcha-version-2-images/samples/samples'))

X = np.zeros((n_samples, 50, 200, 1)) #1070*50*200

y = np.zeros((5, n_samples, 36)) #5*1070*36



for i, pic in enumerate(os.listdir('../input/captcha-version-2-images/samples/samples')):

    # Read image as grayscale

    img_gray = cv2.imread(os.path.join('../input/captcha-version-2-images/samples/samples', pic), cv2.IMREAD_GRAYSCALE)

    pic_target = pic[:-4]#to remove .png

    #the len of pic_target=5

    if len(pic_target) < 6:

        # Scale and reshape image

        img = img_gray / 255.0

        img = np.reshape(img, (50, 200, 1))#1 dimension array

        # Define targets and code them using OneHotEncoding

        targs = np.zeros((5, 36))

        for j, letter in enumerate(pic_target):

            ind = symbols.find(letter)

            targs[j, ind] = 1#one hot encoding

       

        X[i] = img #assign each pixel scaled to the position

        y[:, i] = targs

X_train, y_train = X[:970], y[:, :970]

X_test, y_test = X[970:], y[:, 970:]


model.summary()
model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30,verbose=1, validation_split=0.2)
def predict(filepath):

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    if img is not None:

        img = img / 255.0

    else:

        print("Not detected");

    res = np.array(model.predict(img[np.newaxis, :, :, np.newaxis]))

    ans = np.reshape(res, (5, 36))

    list = []

    for a in ans:

        list.append(np.argmax(a))

        



    capt = ''

    for l in list:

        capt += symbols[l]

    return capt

    

print(model.metrics_names)
score= model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]])

print('Test Loss and accuracy:', score)
print(predict('../input/captcha-version-2-images/samples/samples/8n5p3.png'))

print(predict('../input/captcha-version-2-images/samples/samples/f2m8n.png'))

print(predict('../input/captcha-version-2-images/samples/samples/dce8y.png'))

print(predict('../input/captcha-version-2-images/samples/samples/3eny7.png'))

print(predict('../input/captcha-version-2-images/samples/samples/npxb7.png'))