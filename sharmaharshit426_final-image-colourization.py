# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# Any results you write to the current directory are saved as output.
#importing all the necessary libraries



#To show progress

from tqdm import tqdm



#For tasks related to image processing

import cv2



#For showing graphs and images

import matplotlib.pyplot as plt

%matplotlib inline



#Deep libraries Import

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPool2D

#Here, we will be converting our images into grayscale and storing them into X. We will add the images as is, without any coversion, to Y

def make_data(X, Y, path, img_size):

    for filename in tqdm(os.listdir(path)):

        img = cv2.imread(os.path.join(path, filename))

        img = img.astype('uint8')

        img = cv2.resize(img, dsize=(img_size, img_size))

        

        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        X.append(grayscale_img)

        

        ab_layer = np.zeros((img_size, img_size, 2))

        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        ab_layer = lab_img[:,:,1:]

        Y.append(ab_layer)

        

        

    return X, Y

            
img_size = 200

X = []

Y = []

X_test = []

Y_test = []

train_path = '../input/flower/flower/Train'

test_path = '../input/flower/flower/Test'



X, Y = make_data(X, Y, train_path, img_size)

X_test, Y_test = make_data(X_test, Y_test, test_path, img_size)
X = np.array(X)

Y = np.array(Y)

X_test = np.array(X_test)

Y_test = np.array(Y_test)

X = X / 255

Y = Y / 255


plt.figure()

fig, ax = plt.subplots(8,2, figsize=(15,20))

num = 0

for i in range(8):

    ax[i][0].imshow(X[num], cmap='gray')

    ax[i][0].set_title('Grayscale Image')

    img = np.zeros((img_size, img_size, 3))

    img[:,:,0] = X[num]

    img[:,:,1:] = Y[num]

    img = img * 255

    img = img.astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    ax[i][1].imshow(img)

    ax[i][1].set_title('Colourized Image')

    plt.tight_layout()

    num+=1

X = X.reshape(731, img_size, img_size, 1)

print(X.shape,Y.shape)
model = Sequential()

model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same',

                 input_shape=(img_size, img_size, 1), use_bias=True))



model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', 

                 use_bias=True))



model.add(Conv2D(filters=2, kernel_size=3, activation='relu', padding='same'))



model.summary()
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
history = model.fit(X, Y, verbose=1, batch_size=32, epochs=50)
plt.figure()

plt.plot(history.history['loss'])



X_test = []

Y_test = []

X_test, Y_test = make_data(X_test, Y_test, test_path, img_size)

X_test = np.array(X_test)

X_test = X_test.reshape(112, img_size, img_size, 1)

X_test = X_test/255
plt.figure()

fig, ax = plt.subplots(8,2, figsize=(15,20))

output = model.predict(X_test)

print(output.shape)

for i in range(8):

    ax[i][0].imshow(X_test[num,:,:,0], cmap='gray')

    ax[i][0].set_title('Grayscale Image')

    img = np.zeros((img_size, img_size, 3))

    img[:,:,0] = X_test[num,:,:,0]

    img[:,:,1:] = output[num]

    img = img * 255

    img = img.astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    ax[i][1].imshow(img)

    ax[i][1].set_title('Colourized Image')

    plt.tight_layout()

    num+=1