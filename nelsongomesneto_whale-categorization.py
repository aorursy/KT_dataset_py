# Imports



import os

import time

from heapq import heappush, heappop

import tensorflow as tf

import pandas as pd

import numpy as np

import copy

import matplotlib.pyplot as plot

import matplotlib.image as mpimage

import seaborn as sn

from random import shuffle

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

folder = "../input/whale-categorization-playground/"
# Loading database



idDict = {}

train = pd.read_csv(folder + "train.csv")

for i in train.iterrows():

    if not train.loc[i[0]][1] in idDict:

        idDict[train.loc[i[0]][1]] = len(idDict)

    train.loc[i[0]][0] = folder + "train/train/" + train.loc[i[0]][0]

    train.loc[i[0]][1] = idDict[train.loc[i[0]][1]]

test = os.listdir(folder + "test/test")

for i in range(len(test)):

    test[i] = folder + "test/test/" + test[i]



x_train, y_train = train.iloc[:, 0], train.iloc[:, 1]



width, height, batchSize, iterations = 150, 150, 10000, 100
# Model definition



model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 2), padding="Same", activation="relu", input_shape=(150, 150, 3)),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Dropout(0.1),

    

        tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="Same", activation="relu", input_shape=(25, 25, 3)),

        tf.keras.layers.MaxPool2D(pool_size=(5, 5)),

        tf.keras.layers.Dropout(0.2),



        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(4251, activation="softmax")

    ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train



for k in range(iterations):

    indexes = list(range(len(x_train)))

    shuffle(indexes)

    i, iterationStartTime = 0, time.time()

    while i < len(indexes):

        batchStartTime = time.time()

        x, y = [], []

        for j in range(i, min(len(indexes), i + batchSize)):

            image = tf.keras.preprocessing.image.load_img(x_train[indexes[j]], target_size=(width, height))

            image = tf.keras.preprocessing.image.img_to_array(image)

            x += [image]

            ans = np.zeros(4251)

            ans[y_train[indexes[j]]] = 1

            y += [ans]

        x, y = np.array(x), np.array(y)

        model.fit(x, y, epochs=5, verbose=True, shuffle=True)

        i += batchSize

        print("\tbatch: %Lg%% - %Lg seconds" % (100 * i / len(indexes), time.time() - batchStartTime))

    print("iteration: %Lg%% - %Lg seconds" % (100 * (k+1) / 100, time.time() - iterationStartTime))
# Prepare submission.csv



y_final = []

pos = 0

for i in range(len(test)):

    image = tf.keras.preprocessing.image.load_img(test[i], target_size=(width, height))

    image = tf.keras.preprocessing.image.img_to_array(image)

    x = np.array([image])

    y = model.predict(x)

    ymap, at = [], 0

    for j in sorted(idDict, key=lambda x: x[1]):

        heappush(ymap, [y[0][at], j])

        if len(ymap) > 5:

            heappop(ymap)

        at += 1

    now = []

    for j in range(5):

        now += [heappop(ymap)[1]]

    y_final += [now]

    if (pos % 1000 == 0):

        print(100 * pos / len(test))

    pos += 1
# Making submission.csv



imageId, y_sub = [], []

for i in range(len(test)):

    imageId += [test[i].split('/')[-1]]

    y_sub += [' '.join(y_final[i])]

submission = pd.DataFrame({"Image": imageId, "Id": y_sub})

submission.to_csv("submission.csv", index=False)