import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout

import cv2

from sklearn.model_selection import train_test_split

import os

import random

import matplotlib.pyplot as plt



print(os.listdir("../input"))

train_csv = pd.read_csv('../input/train.csv').sample(frac=1).reset_index(drop=True)

images = train_csv.id.values.tolist() #some bug causes .values to not be accepted as np array

target = train_csv.has_cactus.values.tolist()

train_X, test_X, train_Y, test_Y = train_test_split(images, target, test_size=0.1, random_state=42)

del train_csv, images, target
def get_image(imname):

    name = os.path.join('../input/train/train', imname)

    img = cv2.imread(name, 1)

    img = cv2.resize(img, (32,32))/255

    return img
batch_img = []

batch_tar = []

val_x = []

val_y = []

for i in range(len(train_X)):

    batch_img.append(np.reshape(get_image(train_X[i]), (32,32,3)))

    batch_tar.append(train_Y[i])

for i in range(len(test_X)):

    val_x.append(np.reshape(get_image(test_X[i]), (32,32,3)))

    val_y.append(test_Y[i])

batch_img, val_x = np.array(batch_img), np.array(val_x)
model = Sequential()

model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,3)))

model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))

model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu'))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(batch_img, batch_tar, validation_data = (val_x, val_y), epochs = 20)
test_list = os.listdir('../input/test/test')

def get_test_image(imname):

    name = os.path.join('../input/test/test', imname)

    img = cv2.imread(name, 1)

    img = cv2.resize(img, (32,32))/255

    return img
test_imgs = []

for i in range(len(test_list)):

    test_imgs.append(np.reshape(get_test_image(test_list[i]), (32,32,3)))

test_imgs = np.array(test_imgs)

pred = model.predict(test_imgs)
pred = [i[0] for i in pred]



res_dict = {'id': test_list, 'has_cactus' : pred}

res_df = pd.DataFrame(res_dict)

res_df.to_csv('result.csv', index=False)