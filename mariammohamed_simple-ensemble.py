

import os

print(os.listdir("../input"))


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import glob

import scipy

import cv2

import random

from sklearn.model_selection import KFold



import keras


train_data = pd.read_csv('../input/train.csv')


positive_examples_indexes = train_data[train_data.has_cactus==1].index

negative_examples_indexes = train_data[train_data.has_cactus==0].index


pos_indexes = positive_examples_indexes.tolist()[:]

random.shuffle(pos_indexes)


k = 5

third = int(len(pos_indexes) / k)

folds_indexes = []

for i in range(k):

    start = i*third

    end = (i+1)*third

    if i == k-1:

        end = len(pos_indexes)

    folds_indexes.append(pos_indexes[start:end])


def image_generator(indexes=None, batch_size = 16, shuffle=True, train=True):

    while True:

        if train:

            temp_indexes = indexes[:]

            temp_indexes.extend(negative_examples_indexes.tolist())

            random.shuffle(temp_indexes)

            random.shuffle(temp_indexes)

        else:

            temp_indexes = indexes[:]

            

        N = int(len(temp_indexes) / batch_size)

       



        # Read in each input, perform preprocessing and get labels

        for i in range(N):

            current_indexes = temp_indexes[i*batch_size: (i+1)*batch_size]

            batch_input = []

            batch_output = [] 

            for index in current_indexes:

                img = mpimg.imread('../input/train/train/' + train_data.id[index])

                batch_input += [img]

                batch_input += [img[::-1, :, :]]

                batch_input += [img[:, ::-1, :]]

                batch_input += [np.rot90(img)]

                

                temp_img = np.zeros_like(img)

                temp_img[:28, :, :] = img[4:, :, :]

                batch_input += [temp_img]

                

                temp_img = np.zeros_like(img)

                temp_img[:, :28, :] = img[:, 4:, :]

                batch_input += [temp_img]

                

                temp_img = np.zeros_like(img)

                temp_img[4:, :, :] = img[:28, :, :]

                batch_input += [temp_img]

                

                temp_img = np.zeros_like(img)

                temp_img[:, 4:, :] = img[:, :28, :]

                batch_input += [temp_img]

                

                batch_input += [cv2.resize(img[2:30, 2:30, :], (32, 32))]

                

                batch_input += [scipy.ndimage.interpolation.rotate(img, 10, reshape=False)]

                

                batch_input += [scipy.ndimage.interpolation.rotate(img, 5, reshape=False)]

                

                for _ in range(11):

                    batch_output += [train_data.has_cactus[index]]

                

            batch_input = np.array( batch_input )

            batch_output = np.array( batch_output )

        

            yield( batch_input, batch_output.reshape(-1, 1) )


def build_model():

    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3)))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Conv2D(64, (3, 3)))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Conv2D(128, (3, 3)))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Conv2D(128, (3, 3)))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Conv2D(256, (3, 3)))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    # model.add(keras.layers.Conv2D(128, (3, 3)))

    # model.add(keras.layers.BatchNormalization())

    # # model.add(keras.layers.Activation('relu'))

    # model.add(keras.layers.LeakyReLU(alpha=0.3))

    # model.add(keras.layers.Conv2D(256, (3, 3)))

    # model.add(keras.layers.BatchNormalization())

    # # model.add(keras.layers.Activation('relu'))

    # model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Conv2D(256, (3, 3)))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Conv2D(512, (3, 3)))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Flatten())

    # model.add(keras.layers.Dense(512, activation='relu'))

    model.add(keras.layers.Dense(100))

    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.LeakyReLU(alpha=0.3))

    model.add(keras.layers.Dense(1, activation='sigmoid'))



    opt = keras.optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    

    return model


k


models = []

for i in range(k):

    print('i =', i)

    model = build_model()

    for j in range(k):

        if i == j:

            continue

        current_pos_indexes = folds_indexes[i]

        model.fit_generator(image_generator(current_pos_indexes), steps_per_epoch= len(current_pos_indexes) / 16, epochs=20)

    models.append(model)


for model in models:

    print(model.evaluate_generator(image_generator(indexes=train_data[15000:].index.tolist()), steps=train_data[15000:].shape[0] / 16))


for model in models:

    for j in range(k):

        current_pos_indexes = folds_indexes[j]

        print(model.evaluate_generator(image_generator(indexes=current_pos_indexes), steps=len(current_pos_indexes) / 16))


test_files = os.listdir('../input/test/test/')


preds = []

for _ in range(len(models)):

    preds.append([])

batch = 40

# all_out = []

for i in range(int(4000/batch)):

    images = []

    for j in range(batch):

        img = mpimg.imread('../input/test/test/'+test_files[i*batch + j])

        images += [img]

    for k2 in range(len(models)):

        model = models[k2]

        out = model.predict(np.array(images))

        preds[k2] += [out]

#     all_out += [out]


all_out = np.array(list(map(lambda x: np.array(x).reshape(-1, 1), preds)))


all_out.shape


all_out[:, :10, :]


all_out = np.mean(all_out, axis=0)


all_out.shape


sub_file = pd.DataFrame(data = {'id': test_files, 'has_cactus': all_out.reshape(-1).tolist()})


sub_file.head()


sub_file.to_csv('sample_submission.csv', index=False)