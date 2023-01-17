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


def image_generator2(batch_size = 16, all_data=True, shuffle=True, train=True, indexes=None):

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
positive_examples = train_data[train_data.has_cactus==1]

negative_examples = train_data[train_data.has_cactus==0]
def augment_img(img):

    batch_input = []

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

    

    return batch_input


def image_generator(batch_size = 8, all_data=True, shuffle=True, train=True, indexes=None):

    while True:

        if indexes is None:

            if train:

                indexes = positive_examples.index.tolist()

                neg_indexes = negative_examples.index.tolist()

                if shuffle:

                    np.random.shuffle(indexes)

                    np.random.shuffle(neg_indexes)

            

        N = int(len(indexes) / (batch_size/2))

        neg_N = int(len(neg_indexes) / (batch_size/2))

       

        j = 0



        # Read in each input, perform preprocessing and get labels

        for i in range(N):

            current_indexes = indexes[i*(batch_size//2): (i+1)*(batch_size//2)]

            current_neg_indexes = neg_indexes[j*(batch_size//2): (j+1)*(batch_size//2)]

            j = (j + 1) % neg_N

            batch_input = []

            batch_output = [] 

            for ind in range(len(current_indexes)):

                index = current_indexes[ind]

                neg_index = current_neg_indexes[ind]

                

                img = mpimg.imread('../input/train/train/' + train_data.id[index])

                batch_input.extend(augment_img(img))

                for _ in range(11):

                    batch_output += [train_data.has_cactus[index]]

                

                neg_img = mpimg.imread('../input/train/train/' + train_data.id[neg_index])

                batch_input.extend(augment_img(neg_img))

                for _ in range(11):

                    batch_output += [train_data.has_cactus[neg_index]]

                

#                 factor = 0.05

#                 new_img = factor*neg_img + (1-factor)*img

#                 batch_input.append(new_img)

#                 batch_output += [factor*train_data.has_cactus[neg_index]+(1-factor)*train_data.has_cactus[index]]

                

#                 factor = 0.95

#                 new_img = factor*neg_img + (1-factor)*img

#                 batch_input.append(new_img)

#                 batch_output += [factor*train_data.has_cactus[neg_index]+(1-factor)*train_data.has_cactus[index]]

            

                

                

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


opt = keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):

    '''

    Wrapper function to create a LearningRateScheduler with step decay schedule.

    '''

    def schedule(epoch):

        return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    

    return keras.callbacks.LearningRateScheduler(schedule)



lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=2)

early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=3)



model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 8, epochs=10, callbacks=[lr_sched, early_stop])
# def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):

#     '''

#     Wrapper function to create a LearningRateScheduler with step decay schedule.

#     '''

#     def schedule(epoch):

#         return initial_lr * (decay_factor ** np.floor(epoch/step_size))

    

#     return keras.callbacks.LearningRateScheduler(schedule)



# lr_sched = step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=2)



# model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 8, epochs=20, callbacks=[lr_sched])
model.evaluate_generator(image_generator2(), steps=train_data.shape[0]//16)
# model.evaluate_generator(image_generator(), steps=train_data.shape[0]//8)


# keras.backend.eval(model.optimizer.lr.assign(0.00001))


# model.fit_generator(image_generator(), steps_per_epoch= train_data.shape[0] / 16, epochs=15)


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

                batch_input += [img[:, ::-1, :]]

                batch_output.append(train_data.has_cactus[index])

            

            batch_input = np.array( batch_input )



            model_pred = model.predict_classes(batch_input)

            for j in range(len(batch_output)):

                if model_pred[j] != batch_output[j]:

                    wrong_ind.append(i*batch_size+j)


len(wrong_ind)


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