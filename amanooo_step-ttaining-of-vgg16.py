import pandas as pd

pd.set_option('display.max_columns', None)

pd.options.display.float_format = '{:.5f}'.format

import numpy as np

np.random.seed(2)



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

sns.set(style='white', context='notebook', palette='deep')

%matplotlib inline



import cv2

from PIL import Image



#from sklearn.model_selection import train_test_split

#import itertools



import keras

from keras.models import Sequential, load_model

from keras.layers import Dense, Flatten, Conv2D

from keras.layers import BatchNormalization

from keras.layers.core import Activation

from keras.optimizers import Adam, SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.applications.vgg19 import VGG19, preprocess_input



import random

import os
DIRin = "../input/"

DIRout = ""

print(os.listdir(DIRin))
trains = pd.read_csv(DIRin + "train.csv")

trains['has_cactus'] = trains['has_cactus'].astype(str)

trains.shape
trains.head(10)
sns.countplot(trains.has_cactus)
trains.has_cactus.value_counts()
# For the test data, create a dataframe similar to 'trains'

tests = os.listdir(DIRin + 'test/test')

tests = pd.DataFrame(tests, columns=['id'])

tests['has_cactus'] = str(0.5)

tests.head()
#===== show_image(randum) =============

#    inS = "train"/"test"

#    inNum : number of display

#======================================

def show_image(inS = 'train', inNum = 10):

    if inS == 'train':

        df = trains

    else:

        df = tests

    fig = plt.figure(figsize=(10, inNum//5 * 2))

    for idx, img in enumerate(np.random.choice(df["id"], inNum)):

        ax = fig.add_subplot(inNum//5, 5, idx+1, xticks=[], yticks=[])

        im = Image.open(DIRin + inS + "/" + inS + "/" + img)

        plt.imshow(im)

        lab = df.loc[df['id'] == img, 'has_cactus'].values[0]

        ax.set_title(f'Label: {lab}')
# sample images

show_image('train', 10)
# split train and val data

train_size = 14000

val_trains = trains[train_size:].copy()

val_trains = val_trains.reset_index(drop=True)



# train data

datagen=ImageDataGenerator(rescale=1./255)

X_train=datagen.flow_from_dataframe(dataframe = trains[:train_size],

                                      directory = DIRin + "train/train",

                                      x_col='id', y_col='has_cactus',

                                      batch_size=50,

                                      seed=42, shuffle=True,

                                      class_mode='binary',

                                      target_size=(32,32))

X_val=datagen.flow_from_dataframe(dataframe = val_trains,

                                    directory = DIRin + "train/train",

                                    x_col='id', y_col='has_cactus',

                                    batch_size=50,

                                    seed=42, shuffle=True,

                                    class_mode='binary',

                                    target_size=(32,32))
# base_model

base_model=VGG16(weights="imagenet",

                 include_top=False,

                 input_shape=(32,32,3))

base_model.trainable = False



base_model.summary()
# Model construction

model = Sequential()

model.add(base_model)

    

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(128, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(1, activation = 'sigmoid'))

    

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, nesterov=True),

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.summary()
# learning rate annealer

reduce_lr = ReduceLROnPlateau(monitor='val_acc',

                              patience=3,

                              verbose=1,

                              factor=0.5,

                              min_lr=1e-6)
epochs1 = 20

history1 = model.fit_generator(generator = X_train,

                              steps_per_epoch = X_train.n//X_train.batch_size,

                              epochs = epochs1,

                              validation_data = X_val,

                              validation_steps = X_val.n//X_val.batch_size,

                              verbose = 2,

                              callbacks = [reduce_lr])

# store model after training

model.save(DIRout+"temp.h5")
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(1,2,figsize=(10, 3))



history_df1 = pd.DataFrame(history1.history)

ax[0].plot(history_df1[['loss', 'val_loss']]), ax[0].legend(['loss', 'val_loss'])

ax[1].plot(history_df1[['acc', 'val_acc']]), ax[1].legend(['acc', 'val_acc'])
# train data predict

datagen_p=ImageDataGenerator(rescale=1./255)

X_train_p=datagen_p.flow_from_dataframe(dataframe=trains,

                                   directory = DIRin + "train/train",

                                   x_col='id', y_col='has_cactus',

                                   batch_size=32, 

                                   seed=42, shuffle=False,

                                   class_mode=None,

                                   target_size=(32,32))



# ompare the predict value(pred_1) with the correct lavel(has_cactus)

P_train = model.predict_generator(X_train_p,

                                  steps = X_train_p.n//X_train_p.batch_size)

pred_1 = pd.DataFrame(P_train,columns=["pred_1"])

trains = pd.concat([trains, pred_1], axis = 1)

trains.head(10)
model=load_model(DIRout+"temp.h5")



model.trainable = True

model.compile(optimizer=SGD(lr=0.00005, momentum=0.9, nesterov=True),

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
epochs2 = 20

history2 = model.fit_generator(generator=X_train,

                              steps_per_epoch = X_train.n//X_train.batch_size,

                              epochs = epochs2,

                              validation_data = X_val,

                              validation_steps = X_val.n//X_val.batch_size,

                              verbose = 2,

                              callbacks = [reduce_lr])
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(1,2,figsize=(10, 3))



history_df2 = pd.DataFrame(history2.history)

history_df2.rename(index = lambda x: x+epochs1, inplace=True)

history_df2.rename(columns={'loss':'loss(step2)', 'acc':'acc(step2)',

                            'val_loss':'val_loss(step2)', 'val_acc':'val_acc(step2)'}, inplace=True)

history_df = pd.concat([history_df1,history_df2],sort=False)



ax[0].plot(history_df[['loss', 'val_loss','loss(step2)', 'val_loss(step2)']])

ax[0].legend(['loss(step1)', 'val_loss(step1)','loss(step2)', 'val_loss(step2)'])

ax[1].plot(history_df[['acc', 'val_acc','acc(step2)', 'val_acc(step2)']])

ax[1].legend(['acc(step1)', 'val_acc(step1)','acc(step2)', 'val_acc(step2)'])
# compare the predict value(pred_2) with the correct lavel(has_cactus)

P_train = model.predict_generator(X_train_p,

                                  steps = X_train_p.n//X_train_p.batch_size)

pred_2 = pd.DataFrame(P_train,columns=["pred_2"])

trains = pd.concat([trains, pred_2], axis = 1)

trains.head(10)
# test data

testgen=ImageDataGenerator(rescale=1./255)

X_test=testgen.flow_from_dataframe(dataframe=tests,

                                   directory = DIRin + "test/test",

                                   x_col='id', y_col='has_cactus',

                                   batch_size=32, 

                                   seed=42, shuffle=False,

                                   class_mode=None,

                                   target_size=(32,32))
# predict

P_test = model.predict_generator(X_test,

                                 steps=X_test.n//X_test.batch_size)
tests['pred'] = P_test

tests['has_cactus'] = tests['pred'].apply(lambda x: 1 if x > 0.5 else 0)

tests.head(15)
# predicted sumple

show_image('test', 15)
submit = tests.drop("pred", axis=1)

submit.to_csv(DIRout + 'solution_01.csv', index=False)