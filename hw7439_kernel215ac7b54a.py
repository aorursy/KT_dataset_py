import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math

import random

import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils import np_utils

from keras.callbacks import EarlyStopping

import os

import cv2

import ast

%matplotlib inline

path_dir = "../input/quickdraw-doodle-recognition/train_simplified"

file_list = os.listdir(path_dir)



file_list = [word.replace('.csv', '') for word in file_list]



file_list
path_csv3 = '../input/quickdraw-doodle-recognition/test_simplified.csv'



data3 = pd.read_csv(path_csv3)



data3['listed'] = "a"

data3['probability'] = 0.0001

data3['listed2'] = "a"

data3['probability2'] = 0.0001

data3['listed3'] = "a"

data3['probability3'] = 0.0001
def draw_matrix(list_raw):

    A = np.zeros((256, 256))

    xx = []

    yy = []

    for list1 in list_raw:

        xx = xx + list1[0]

        yy = yy + list1[1]

    minx = min(xx)

    maxx = max(xx)

    miny = min(yy)

    maxy = max(yy)



    midx = round(127-(maxx-minx)/2)

    midy = round(127-(maxy-miny)/2)

    

    for i in range(len(list_raw)):

        length1 = len(list_raw[i][0])

        length2 = len(list_raw[i][1])

        list_raw[i][0] = [min(list_raw[i][0][j]+midx, 255) for j in range(length1)]

        list_raw[i][1] = [min(list_raw[i][1][j]+midy, 255) for j in range(length2)]

        

    for list1 in list_raw:

        for i in range(1,len(list1[0])):

            x2 = list1[0][i]

            y2 = list1[1][i]

            x1 = list1[0][i-1]

            y1 = list1[1][i-1]

            

            decide = max(abs(x2-x1), abs(y2-y1))

            if decide == abs(x2- x1) and decide > 0:

                slope = (y2-y1)/(x2-x1)

                if x1 < x2:

                    for j in range(x1, x2+1):

                        x = j

                        y = y1+slope*(j-x1)

                        y = round(y)

                        A[x, y] = 1

                else:

                    for j in range(x2, x1+1):

                        x = j

                        y = y1+slope*(j-x1)

                        y = round(y)

                        A[x, y] = 1

            elif decide == abs(y2-y1) and decide > 0:

                slope = (x2-x1)/(y2-y1)

                if y1 < y2:

                    for j in range(y1, y2+1):

                        y = j

                        x = x1+slope*(j-y1)

                        x = round(x)

                        A[x, y] = 1                   

                else:

                    for j in range(y2, y1+1):

                        y = j

                        x = x1+slope*(j-y1)

                        x = round(x)

                        A[x, y] = 1           

            elif x1 == x2:

                if y1 < y2:

                    for j in range(y1, y2+1):

                        A[x1, j] = 1

                else:

                    for j in range(y2, y1+1):

                        A[x1, j] = 1

            elif y1 == y2:

                if x1 < x2:

                    for j in range(x1, x2+1):

                        A[j, y1] = 1

                else:

                    for j in range(x2, x1+1):

                        A[j, y1] = 1

                        

    return A

data_example = pd.read_csv(path_csv3)

list_raw = data_example['drawing'][100]

list_raw = ast.literal_eval(list_raw)

A = draw_matrix(list_raw)

plt.matshow(A)
model = Sequential()

model.add(Conv2D(100, kernel_size=(5, 5), strides=(1, 1), padding='same',

                 activation='relu',

                 input_shape=(256, 256,1)))



model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

model.add(Conv2D(64, (4, 4), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(4, 4)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(17, activation='softmax'))

model.summary()
file_list = random.sample(file_list, 340)



train_N = 600

test_N = 200





for epoch in range(20):

    A = np.zeros((17*train_N, 256, 256, 1))

    B = np.zeros((17*test_N, 256, 256, 1))

    A1 = [i for i in range(17*train_N)]

    sampling2 = random.sample(A1, 17*train_N)

    B1 = [i for i in range(17*test_N)]

    sampling3 = random.sample(B1, 17*test_N)



    y_train = [0 for i in range(17*train_N)]

    y_test = [0 for i in range(17*test_N)]

    l_train = 0

    l_test = 0

    l = 0

    

    file_list2 = file_list[(17*epoch):(17*epoch+17)]

    for download in file_list2:

        path_csv = '../input/quickdraw-doodle-recognition/train_simplified/'

        path_csv2 = path_csv + download + '.csv'

        data = pd.read_csv(path_csv2)

        select = [i for i in range(len(data['drawing']))]

        sampling = random.sample(select, train_N+test_N)

        for i in range(train_N):

            key = sampling[i]

            list_raw = data['drawing'][key]

            list_raw = ast.literal_eval(list_raw)

            A[sampling2[l_train], :, :, 0] = draw_matrix(list_raw)

            y_train[sampling2[l_train]] = l

            l_train += 1

        

        

        for i in range(train_N, test_N):

            key = sampling[i]

            list_raw = data['drawing'][key]

            list_raw = ast.literal_eval(list_raw)

            B[sampling3[l_test], :, :, 0] = draw_matrix(list_raw)

            y_test[sampling3[l_test]] = l

            l_test += 1

        l += 1

    y_train = np_utils.to_categorical(y_train, 17)

    y_test = np_utils.to_categorical(y_test, 17)

    

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    batch_size = 50

    epochs = 5

    x_train = A

    x_test = B

    

    hist = model.fit(x_train, y_train, validation_data = (x_test, y_test),

                 batch_size=batch_size,

                 epochs=epochs,

                 verbose=1)

    

    

    for testing in range(len(data3)):

        K = np.zeros((256, 256))

        list_raw = data3['drawing'][testing]

        list_raw = ast.literal_eval(list_raw)

        K = np.zeros((1, 256, 256, 1))

        K[0, :, :,0] = draw_matrix(list_raw)

        

        predicted_result = model.predict(K)

        key = np.where(predicted_result[0] == max(predicted_result[0]))

        key2 = list(key[0])[0]

        name = file_list2[key2]

        probs = max(predicted_result[0])

        if probs > data3['probability'][testing]:

            if data3['probability2'][testing] == 0.0001:

                data3['listed2'][testing] = name

                data3['probability2'][testing] = probs

            elif data3['probability3'][testing] == 0.0001:

                data3['listed3'][testing] = name

                data3['probability3'][testing] = probs

            else:

                data3['listed'][testing] = name

                data3['probability'][testing] = probs

        elif probs >data3['probability2'][testing]:

            data3['listed2'][testing] = name

            data3['probability2'][testing] = probs

        elif probs >data3['probability3'][testing]:

            data3['listed3'][testing] = name

            data3['probability3'][testing] = probs
data_final = pd.read_csv('../input/quickdraw-doodle-recognition/sample_submission.csv')



for i in range(len(data3)):

    data3['listed'][i] = data3['listed'][i].replace(" ", "_")

    data3['listed2'][i] = data3['listed2'][i].replace(" ", "_")

    data3['listed3'][i] = data3['listed3'][i].replace(" ", "_")

    data_final['word'][i] = data3['listed'][i] + " " + data3['listed2'][i] + " " + data3['listed3'][i]

    if i % 1000 == 0:

        print(i//1000)
data_final.to_csv('submission_final.csv', index = False)