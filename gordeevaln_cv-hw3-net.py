# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import skimage.io

from skimage.transform import resize

import matplotlib.pyplot as plt

from tqdm import tqdm

import os
path_to_train = '../input/tl-signs-hse-itmo-2020-winter/train/train/'

data = pd.read_csv('../input/tl-signs-hse-itmo-2020-winter/train.csv')



train_dataset_info = []

for name, label in zip(data['filename'], data['class_number']):

    train_dataset_info.append({

        'path':os.path.join(path_to_train, name),

        'label':int(label) - 1})

train_dataset_info = np.array(train_dataset_info)
train_dataset_info.shape
class data_generator:

    """

    Генератор случайного батча из данного набора данных

    """

    def create_train(dataset_info, batch_size, shape):

        while True:

            random_indexes = np.random.choice(len(dataset_info), batch_size)

            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))

            batch_labels = np.zeros((batch_size, 66))

            for i, idx in enumerate(random_indexes):

                image = data_generator.load_image(

                    dataset_info[idx]['path'], shape)

                batch_images[i] = image

                batch_labels[i][dataset_info[idx]['label']] = 1

            yield batch_images, batch_labels

            

    

    def load_image(path, shape):

        image = skimage.io.imread(path)

        image = resize(image, (shape[0], shape[1]), mode='reflect')

        return image
# create train datagen

train_datagen = data_generator.create_train(

    train_dataset_info, 5, (48, 48, 3))
images, labels = next(train_datagen)



fig, ax = plt.subplots(1, 5, figsize = (25, 5))

for i in range(5):

    ax[i].imshow(images[i])

print('min: {0}, max: {1}'.format(images.min(), images.max()))
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import tensorflow as tf

import keras



def create_model(input_shape, n_out):

    

#     pretrain_model = InceptionResNetV2(

#         include_top=False, 

#         weights='imagenet', 

#         input_shape=input_shape)

    

    model = Sequential([

        Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,

                           input_shape=(48, 48, 3)),

        MaxPooling2D((2, 2), strides=2),

        Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),

        MaxPooling2D((2, 2), strides=2),

        Flatten(),

        Dense(128, activation=tf.nn.relu),

        Dense(66,  activation=tf.nn.softmax)

    ])

    return model
keras.backend.clear_session()



model = create_model(

    input_shape=(48, 48, 3), 

    n_out=66)



model.compile(

    loss='categorical_crossentropy', 

    optimizer=Adam(1e-04),

    metrics=['acc'])

model.summary()
epochs = 100; batch_size = 16

checkpointer = ModelCheckpoint(

    '../working/InceptionResNetV2.model', 

    verbose=2, 

    save_best_only=True)



# split and suffle data 

np.random.seed(2018)

indexes = np.arange(train_dataset_info.shape[0])

np.random.shuffle(indexes)

train_indexes = indexes[:20000]

valid_indexes = indexes[20000:]



# create train and valid datagens

train_generator = data_generator.create_train(

    train_dataset_info[train_indexes], batch_size, (48, 48, 3))

validation_generator = data_generator.create_train(

    train_dataset_info[valid_indexes], 100, (48, 48, 3))



# train model

history = model.fit_generator(

    train_generator,

    steps_per_epoch=100,

    validation_data=next(validation_generator),

    epochs=epochs, 

    verbose=1,

    callbacks=[checkpointer])
fig, ax = plt.subplots(1, 2, figsize=(15,5))

ax[0].set_title('loss')

ax[0].plot(history.epoch, history.history["loss"], label="Train loss")

ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")

ax[1].set_title('acc')

ax[1].plot(history.epoch, history.history["acc"], label="Train acc")

ax[1].plot(history.epoch, history.history["val_acc"], label="Validation acc")

ax[0].legend()

ax[1].legend()
submit = pd.read_csv('../input/tl-signs-hse-itmo-2020-winter/sample_submission.csv')



predicted = []

for name in tqdm(submit['filename']):

    path = os.path.join('../input/tl-signs-hse-itmo-2020-winter/test/test', name)

    image = data_generator.load_image(path, (48, 48, 3))

    score_predict = model.predict(image[np.newaxis])[0]

    label_predict = np.argmax(score_predict)# np.arange(66)[score_predict>=0.5]

    str_predict_label = str(label_predict + 1)

    predicted.append(str_predict_label)
submit['class_number'] = predicted

submit.to_csv('submission_neural.csv', index=False)
score_predict
label_predict