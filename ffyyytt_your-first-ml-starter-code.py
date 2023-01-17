import numpy as np

import pandas as pd

import cv2

import os



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras.preprocessing import image



import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from tqdm import tqdm
TrainImagePaths = []

for dirname, _, filenames in os.walk('/kaggle/input/shopee-product-detection-student/train/train/train'):

    for filename in filenames:

        if (filename[-3:] == 'jpg'):

            TrainImagePaths.append(os.path.join(dirname, filename))
imgSize = 28

X_train = []

Y_train = []

for imagePath in tqdm(TrainImagePaths):

    label = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (imgSize, imgSize))



    X_train.append(image)

    Y_train.append(int(label))

    

X_train = np.array(X_train).astype('float16')/255

Y_train = to_categorical(Y_train)
model = Sequential()

model.add(Conv2D(8, kernel_size=(3, 3),activation='relu',input_shape=(imgSize,imgSize,3)))

model.add(Flatten())

model.add(Dense(42, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
df = pd.read_csv('../input/shopee-product-detection-student/test.csv')

X_test = []

for imageName in tqdm(df['filename']): 

    image = cv2.imread('../input/shopee-product-detection-student/test/test/test/'+imageName)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (imgSize, imgSize))

    X_test.append(image)

X_test = np.array(X_test).astype('float16')/255
res = model.predict(X_test, batch_size=32)

res = np.argmax(res, axis=1)

df['category'] = res

df['category'] = df.category.apply(lambda c: str(c).zfill(2))

df.to_csv('output.csv', index = False)