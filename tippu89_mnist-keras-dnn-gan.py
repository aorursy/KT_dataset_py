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
# importing libraries



import warnings

import scipy

import datetime

import argparse

import math

import timeit



from scipy import misc

from skimage import io

from PIL import Image



warnings.filterwarnings("ignore")
# importing keras libraries



import keras

from keras import utils

from keras import models

from keras import backend as K

from keras import optimizers

from keras.layers.core import (Dense, Dropout, Activation, Flatten, Reshape)

from keras.layers.convolutional import (Conv2D, MaxPooling2D)



from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import UpSampling2D
# loading the mnist data



df = pd.read_csv("../input/train.csv")
df.head()
df.info()
df.describe()
# loading the testing data



test = pd.read_csv("../input/test.csv")
test.head()
test.info()
test.describe()
# misc properties



BATCH_SIZE = 64 # number of images per batch

NP_EPOCHS = 10 # number of times the data has been projected towards the model

NP_CLASSES = 10 # number of output classes has been passed to the network

VERBOSE = 1 # verbose logs

VALIDATION_SPLIT = 0.20 # splitting the data into train and validation

OPTIMIZER = optimizers.RMSprop() # optimizer has been passed to the model
# dividing the data into train and target 



train = df.drop(['label'], axis=1)

target = df['label']
train.head()
target.head()
# import seaborn and matplotlib



%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
sns.distplot(target)
sns.countplot(target)
# normalizing the train data

train /= 255



# normalizing the test data

test /= 255
# reshaping the train data



train = train.values.reshape(-1, 28, 28, 1)
# reshaping the test data



test = test.values.reshape(-1, 28, 28, 1)
# Label Encoding

from keras.utils import np_utils



target = np_utils.to_categorical(target, NP_CLASSES)
target[:5]
# some examples



plt.imshow(train[42][:, :, 0])
from keras.preprocessing import image

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, BatchNormalization, Activation, Dropout, LeakyReLU

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import optimizers



NB_OUTPUT_FUNC = "softmax"

NB_DESCRIMINATOR_FUNC="sigmoid"

DROPOUT_FIRST = 0.25

DROPOUT_SECOND = 0.20
model = Sequential()

model.add(Conv2D(32, (5, 5), padding='same', input_shape=(28, 28, 1)))

model.add(LeakyReLU(alpha=0.02))

model.add(Conv2D(32, (5, 5)))

model.add(LeakyReLU(alpha=0.02))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(DROPOUT_FIRST))



model.add(Conv2D(64, (3, 3), padding='same'))

model.add(LeakyReLU(alpha=0.02))

model.add(Conv2D(64, (3, 3)))

model.add(LeakyReLU(alpha=0.02))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(DROPOUT_FIRST))



model.add(Flatten())

model.add(Dense(128))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_SECOND))



model.add(Dense(128))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(DROPOUT_SECOND))



model.add(Dense(NP_CLASSES))

model.add(Activation(NB_OUTPUT_FUNC))
model.summary()
current_dt_time = datetime.datetime.now()

model_name = 'model_init' + '_' + str(current_dt_time).replace(' ', '').replace(':', '_') + '/'



if not os.path.exists(model_name):

    os.mkdir(model_name)

    

file_path = model_name + "model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5"

checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,

                             save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.000001, verbose=1, cooldown=1)

callbacks = [checkpoint, LR]
# creating optimizer



# optimizer = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)
# compiling the model



model.compile(optimizer=optimizer, 

              loss='categorical_crossentropy', 

              metrics=['categorical_accuracy'])
# splitting the data for training and validation

import sklearn

from sklearn import model_selection



X_train, X_val, y_train, y_val = model_selection.train_test_split(train, target, test_size=0.20, random_state=123456789)
print("X training data shape: {}".format(X_train.shape))

print("Y training data shape: {}".format(y_train.shape))
print("X validation data shape: {}".format(X_val.shape))

print("Y validation data shape: {}".format(y_val.shape))
# creating a generator to pull the data as batches in a lazy format - Data augumentation



generator = image.ImageDataGenerator(

    featurewise_center=True,

    featurewise_std_normalization=True,

    rotation_range=10,

    zoom_range=0.1,

    width_shift_range=0.1,

    height_shift_range=0.1,

    vertical_flip=False,

    horizontal_flip=False)
num_train_sequences = len(X_train)

num_val_sequences = len(X_val)



print("# training sequences: {}".format(num_train_sequences))

print("# validation sequences: {}".format(num_val_sequences))
# calculating number of training and validation steps per epoch

# for training

if (num_train_sequences % BATCH_SIZE) == 0:

    steps_per_epoch = int(num_train_sequences / BATCH_SIZE)

else:

    steps_per_epoch = int(num_train_sequences / BATCH_SIZE) + 1

    

# for validation    

if (num_val_sequences % BATCH_SIZE) == 0:

    validation_steps = int(num_val_sequences / BATCH_SIZE)

else:

    validation_steps = int(num_val_sequences / BATCH_SIZE) + 1    

    

print("# number of steps required for training: {}".format(steps_per_epoch))

print("# number of steps required for validation: {}".format(validation_steps))
# fitting the model



history = model.fit_generator(generator.flow(X_train, y_train, batch_size=BATCH_SIZE), 

                             validation_data=generator.flow(X_val, y_val, batch_size=BATCH_SIZE),

                             epochs=NP_EPOCHS,

                             verbose=VERBOSE,

                             steps_per_epoch=steps_per_epoch,

                             validation_steps=validation_steps,

                             class_weight=None,

                             initial_epoch=0,

                             callbacks=callbacks)
# best accuracy found

# best model: saving model to model_init_2019-07-0108_08_42.188814/model-00009-0.21558-0.93310-0.12834-0.96155.h5



values = {}

models = os.listdir(model_name)



for model in models:

    converted = model.replace(".h5", "")

    accuracy = float(converted.split("-")[-1])

    values.update({accuracy: model})

    

key = max(values, key = values.get)

best = values.get(key)



print("Best model found: {}".format(best))
# all data in history



history.history.keys()
plt.plot(history.history['categorical_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
# loading the best model

from keras.models import load_model



best_model_path = model_name + best

print("Full path found: {}".format(best_model_path))



best_model = load_model(best_model_path)
best_model.summary()
results = best_model.predict(test)
results[:5]
convert = np.argmax(results, axis=1)
results = pd.Series(convert, name='Label')
results.head()
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"), results],axis = 1)
submission.head()
submission.to_csv("submissions-10epochs.csv", index=False)
from keras import backend as K



K.set_image_data_format("channels_first")
## creating the DCGAN - generator model



def generator():

    

    gen = Sequential()

    gen.add(Dense(input_dim=100, output_dim=1024))

    gen.add(LeakyReLU(alpha=0.02))

    gen.add(Dense(128 * 7 * 7))

    gen.add(BatchNormalization())

    gen.add(LeakyReLU(alpha=0.02))

    gen.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))

    gen.add(UpSampling2D(size=(2,2)))

    gen.add(Conv2D(64, 5, 5, border_mode="same"))

    gen.add(LeakyReLU(alpha=0.02))

    gen.add(UpSampling2D(size=(2,2)))

    gen.add(Conv2D(1, 5, 5, border_mode="same"))

    gen.add(LeakyReLU(alpha=0.02))

    return gen
generator().summary()
## creating the DCGAN - descriminator model



def descriminator():

    

    des = Sequential()    

    des.add(Conv2D(64, (5, 5), padding='same', input_shape=(1, 28, 28)))

    des.add(LeakyReLU(alpha=0.02))

    des.add(MaxPooling2D(pool_size=(2, 2)))

    des.add(Conv2D(128, (5, 5)))

    des.add(LeakyReLU(alpha=0.02))

    des.add(MaxPooling2D(pool_size=(2, 2)))

    des.add(Flatten())

    des.add(Dense(1024))

    des.add(LeakyReLU(alpha=0.02))

    des.add(Dense(1))

    des.add(Activation(NB_DESCRIMINATOR_FUNC))

        

    return des
descriminator().summary()
# changing the number of epochs



NP_EPOCHS = 100
# combining the images



def combine(generated_images):

    number = generated_images.shape[0]

    width = int(math.sqrt(number))

    height = int(math.ceil(float(number) / width))

    shape = generated_images.shape[2:]

    image = np.zeros((height * shape[0], width * shape[1]), dtype=generated_images.dtype)

    

    for index, img in enumerate(generated_images):

        i = int(index / width)

        j = index % width

        

        image[i * shape[0]:(i+1) * shape[0], j * shape[1]:(j+1) * shape[1]] = img[0, :, :]

    

    return image
# generator containing descriminator



def generator_containing_desciminator(generator, descriminator):

    model = Sequential()

    model.add(generator)

    descriminator.trainable = False

    model.add(descriminator)

    return model
# training the network



def train():

    

    # descriminator model

    descriminator_model = descriminator()   

    # generator model

    generator_model = generator()

    

    # descriminator on generator

    descriminator_on_generator = generator_containing_desciminator(generator_model, descriminator_model)

    

    # descriminator optimizer

    descriminator_optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)

    # generator optimizer

    generator_optimizer =  optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)

    

    # compiling the model

    generator_model.compile(loss="binary_crossentropy", optimizer=generator_optimizer)

    descriminator_on_generator.compile(loss="binary_crossentropy", optimizer=generator_optimizer)

    

    descriminator.trainable = True    

    descriminator_model.compile(loss="binary_crossentropy", optimizer=descriminator_optimizer)

        

    noise = np.zeros((BATCH_SIZE, NP_EPOCHS))

    

    for epoch in range(NP_EPOCHS):

        

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):

            for i in range(BATCH_SIZE):

                noise[i, :] = np.random.uniform(-1, 1, 100)

                

            image_batch = X_train[index * BATCH_SIZE: (index + 1) * BATCH_SIZE]

            generated_images = generator_model.predict(noise, verbose=0)

            

            if index % 20 == 0:

                image = combine(generated_images)

                image = image * 127.5 + 127.5

                Image.fromarray(image.astype(np.uint8)).save(str(epoch) + "_" + str(index) + ".png")

                

            X = np.concatenate((image_batch, generated_images))

            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            descriminator_loss = descriminator_model.train_on_batch(X, y)

            print("Batch: {} found loss: {}".format(index, descriminator_loss))

            

            for i in range(BATCH_SIZE):

                noise[i, :] = np.random.uniform(-1, 1, 100)

            descriminator_model.trainable = False

            

            generator_loss = descriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)

            print("Batch: {} found loss: {}", (index, generator_loss))

            

            if index % 10 == 9:

                generator_model.save_weights("generator", True)

                descriminator_model.save_weights("descriminator", True)    
# calling the training method



train()
X_train.shape
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = (X_train.astype(np.float32) - 127.5)/127.5

X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])