# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import glob

import scipy

import cv2



import keras


import random


train_data = pd.read_csv('../input/train.csv')


train_data.shape


train_data.head()


train_data.has_cactus.unique()


train_data.has_cactus.hist()


train_data.has_cactus.value_counts()


train_data.has_cactus.plot()


def image_generator(batch_size = 16, all_data=True, shuffle=True, train=True, indexes=None):

    while True:

        if indexes is None:

            if train:

                if all_data:

                    indexes = np.arange(train_data.shape[0])

                else:

                    indexes = np.arange(train_data[:15000].shape[0])

                if shuffle:

                    np.random.shuffle(indexes)

            else:

                indexes = np.arange(train_data[15000:].shape[0])

            

        N = int(len(indexes) / batch_size)

       



        # Read in each input, perform preprocessing and get labels

        for i in range(N):

            current_indexes = indexes[i*batch_size: (i+1)*batch_size]

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
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, (5, 5), input_shape=(32, 32, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Conv2D(64, (5, 5)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Conv2D(128, (5, 5)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Conv2D(128, (5, 5)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Conv2D(256, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Conv2D(256, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Conv2D(512, (3, 3)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Flatten())





model.add(keras.layers.Dense(100))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(keras.layers.Dense(1, activation='sigmoid'))


model.summary()


opt = keras.optimizers.Adam(0.0001)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 16, epochs=30)


keras.backend.eval(model.optimizer.lr.assign(0.00001))


model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 16, epochs=15)


indexes = np.arange(train_data.shape[0])

N = int(len(indexes) / 64)   

batch_size = 64



wrong_ind = []

for i in range(N):

            current_indexes = indexes[i*64: (i+1)*64]

            batch_input = []

            batch_output = [] 

            for index in current_indexes:

                img = mpimg.imread('../input/train/train/' + train_data.id[index])

                batch_input += [img]

                batch_output.append(train_data.has_cactus[index])

            

            batch_input = np.array( batch_input )

#             batch_output = np.array( batch_output )



            model_pred = model.predict_classes(batch_input)

            for j in range(len(batch_output)):

                if model_pred[j] != batch_output[j]:

                    wrong_ind.append(i*batch_size+j)


len(wrong_ind)


indexes = np.arange(train_data.shape[0])

N = int(len(indexes) / 64)   

batch_size = 64



wrong_ind = []

for i in range(N):

            current_indexes = indexes[i*64: (i+1)*64]

            batch_input = []

            batch_output = [] 

            for index in current_indexes:

                img = mpimg.imread('../input/train/train/' + train_data.id[index])

                batch_input += [img[::-1, :, :]]

                batch_output.append(train_data.has_cactus[index])

            

            batch_input = np.array( batch_input )



            model_pred = model.predict_classes(batch_input)

            for j in range(len(batch_output)):

                if model_pred[j] != batch_output[j]:

                    wrong_ind.append(i*batch_size+j)


wrong_ind


indexes = np.arange(train_data.shape[0])

N = int(len(indexes) / 64)   

batch_size = 64



wrong_ind = []

for i in range(N):

            current_indexes = indexes[i*64: (i+1)*64]

            batch_input = []

            batch_output = [] 

            for index in current_indexes:

                img = mpimg.imread('../input/train/train/' + train_data.id[index])

                batch_input += [img[:, ::-1, :]]

                batch_output.append(train_data.has_cactus[index])

            

            batch_input = np.array( batch_input )



            model_pred = model.predict_classes(batch_input)

            for j in range(len(batch_output)):

                if model_pred[j] != batch_output[j]:

                    wrong_ind.append(i*batch_size+j)


wrong_ind


!ls ../input/test/test/* | wc -l


test_files = os.listdir('../input/test/test/')


len(test_files)


batch = 40

all_out = []

for i in range(int(4000/batch)):

    images = []

    for j in range(batch):

        img = mpimg.imread('../input/test/test/'+test_files[i*batch + j])

        images += [img]

    out = model.predict(np.array(images))

    all_out += [out]


all_out = np.array(all_out).reshape((-1, 1))



all_out.shape


sub_file = pd.DataFrame(data = {'id': test_files, 'has_cactus': all_out.reshape(-1).tolist()})


sub_file.to_csv('sample_submission.csv', index=False)