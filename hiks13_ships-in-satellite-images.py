# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD

from matplotlib import pyplot as plt

from PIL import Image, ImageDraw 





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import json

f = open(r'../input/shipsnet.json')

dataset = json.load(f)

f.close()
input_data = np.array(dataset['data']).astype('uint8')

output_data = np.array(dataset['labels']).astype('uint8')
input_data.shape,output_data.shape
X = input_data.reshape([-1,3, 80, 80])

X[0].shape
pic = X[0]



rad_spectrum = pic[0]

green_spectrum = pic[1]

blue_spectum = pic[2]



plt.figure(2, figsize = (5*3, 5*1))

plt.set_cmap('jet')



# show each channel

plt.subplot(1, 3, 1)

plt.imshow(rad_spectrum)



plt.subplot(1, 3, 2)

plt.imshow(green_spectrum)



plt.subplot(1, 3, 3)

plt.imshow(blue_spectum)

    

plt.show()
output_data.shape
out_y = to_categorical(output_data, num_classes = 2)
indexes = np.arange(2800)

np.random.shuffle(indexes)

X_train = X[indexes].transpose([0,2,3,1])

y_train = out_y[indexes]
X_train = X_train / 255

np.random.seed(2)
model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5, 5), activation='relu',

                 input_shape = (80, 80, 3)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (4, 4), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(3,3)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))
model.summary()
from keras.optimizers import Adamax

optimizer = Adamax(lr=0.001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])


# training

model.fit(

    X_train, 

    y_train,

    batch_size=32,

    epochs=2,

    validation_split=0.2,

    shuffle=True,

    verbose=2

    )
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

                            rotation_range = 90,

                             zoom_range = 0.3,

                            )



datagen.fit(X_train)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state= 3)
from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.fit_generator(datagen.flow(X_train,y_train, batch_size= 86),shuffle = True,

                                  epochs = 10, validation_data = (X_val,y_val),

                                  verbose = 2, steps_per_epoch=X_train.shape[0] // 86,

                                  callbacks=[learning_rate_reduction])
image = Image.open('../input/scenes/scenes/sfbay_2.png')

pix = image.load()

n_spectrum = 3

width = image.size[0]

height = image.size[1]

picture_vector = []

for chanel in range(n_spectrum):

    for y in range(height):

        for x in range(width):

            picture_vector.append(pix[x, y][chanel])



picture_vector = np.array(picture_vector).astype('uint8')

picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)
plt.figure(1, figsize = (15, 30))



plt.subplot(3, 1, 1)

plt.imshow(picture_tensor)



plt.show()
picture_tensor = picture_tensor.transpose(2,0,1)
def cutting(x, y):

    area_study = np.arange(3*80*80).reshape(3, 80, 80)

    for i in range(80):

        for j in range(80):

            area_study[0][i][j] = picture_tensor[0][y+i][x+j]

            area_study[1][i][j] = picture_tensor[1][y+i][x+j]

            area_study[2][i][j] = picture_tensor[2][y+i][x+j]

    area_study = area_study.reshape([-1, 3, 80, 80])

    area_study = area_study.transpose([0,2,3,1])

    area_study = area_study / 255

    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))

    return area_study
def not_near(x, y, s, coordinates):

    result = True

    for e in coordinates:

        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:

            result = False

    return result
def show_ship(x, y, acc, thickness=5):   

    for i in range(80):

        for ch in range(3):

            for th in range(thickness):

                picture_tensor[ch][y+i][x-th] = -1



    for i in range(80):

        for ch in range(3):

            for th in range(thickness):

                picture_tensor[ch][y+i][x+th+80] = -1

        

    for i in range(80):

        for ch in range(3):

            for th in range(thickness):

                picture_tensor[ch][y-th][x+i] = -1

        

    for i in range(80):

        for ch in range(3):

            for th in range(thickness):

                picture_tensor[ch][y+th+80][x+i] = -1
import sys

step = 10; coordinates = []

for y in range(int((height-(80-step))/step)):

    for x in range(int((width-(80-step))/step) ):

        area = cutting(x*step, y*step)

        result = model.predict(area)

        if result[0][1] > 0.90 and not_near(x*step,y*step, 88, coordinates):

            coordinates.append([[x*step, y*step], result])

            print(result)

            plt.imshow(area[0])

            plt.show()
for e in coordinates:

    show_ship(e[0][0], e[0][1], e[1][0][1])
picture_tensor = picture_tensor.transpose(1,2,0)

picture_tensor.shape
plt.figure(1, figsize = (15, 30))



plt.subplot(3,1,1)

plt.imshow(picture_tensor)



plt.show()
