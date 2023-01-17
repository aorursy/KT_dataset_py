import os

import pandas as pd

import numpy as np

import tensorflow as tf

import json

import cv2

import matplotlib.pyplot as plt

import datetime as dt

from tqdm import tqdm

from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import MobileNet

from tensorflow.keras.applications.mobilenet import preprocess_input
DP_DIR = '../input/shuffle-csvs/'

INPUT_DIR = '../input/quickdraw-doodle-recognition/'

NCSVS = 100

NCATS = 340

BASE_SIZE = 256

size = 64

epochs = 30

batch_size = 100

start = dt.datetime.now()
def draw_img(lines):

    img = np.zeros((BASE_SIZE, BASE_SIZE))

    for line in lines:

        for i in range(len(line[0]) - 1):

            _ = cv2.line(img, (line[0][i], line[1][i]), (line[0][i + 1], line[1][i + 1]), 255, 6)

    return cv2.resize(img, (size, size))
def image_gen(batchsize, cnt):

    while True:

        for k in np.random.permutation(cnt):

            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))

            for df in pd.read_csv(filename, chunksize=batchsize):

                df['drawing'] = df['drawing'].apply(json.loads)

                x = np.zeros((len(df), size, size, 1))

                for i, lines in enumerate(df.drawing.values):

                    x[i, :, :, 0] = draw_img(lines)

                    

                x = preprocess_input(x).astype(np.float32)

                y = keras.utils.to_categorical(df.y, num_classes=NCATS)

                

                yield x, y
def df_to_image(df):

    df['drawing'] = df['drawing'].apply(json.loads)

    x = np.zeros((len(df), size, size, 1))

    for i, lines in enumerate(df.drawing.values):

        x[i, :, :, 0] = draw_img(lines)

    x = preprocess_input(x).astype(np.float32)

    return x
train_datagen = image_gen(batch_size, range(NCSVS - 1))
files = sorted(os.listdir('../input/quickdraw-doodle-recognition/train_simplified/'), reverse=False, key=str.lower)

class_dict = {file[:-4].replace(" ", "_"): i for i, file in enumerate(files)}

classreverse_dict = {v: k for k, v in class_dict.items()}
def CNN_model():

    model = MobileNet(input_shape=(size, size, 1), alpha=1., weights=None, classes=NCATS)

    

#     My Own Model

#     model = Sequential()



#     model.add(Conv2D(32,kernel_size=3,activation='relu',padding='same',input_shape=(size,size,1)))

#     model.add(BatchNormalization())

#     model.add(Conv2D(32,kernel_size=3,activation='relu', padding='same'))

#     model.add(BatchNormalization())

#     model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

#     model.add(BatchNormalization())

#     model.add(Dropout(0.4))



#     model.add(Conv2D(64,kernel_size=3,activation='relu', padding='same'))

#     model.add(BatchNormalization())

#     model.add(Conv2D(64,kernel_size=3,activation='relu', padding='same'))

#     model.add(BatchNormalization())

#     model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

#     model.add(BatchNormalization())

#     model.add(Dropout(0.4))



#     model.add(Flatten())

#     model.add(Dense(2 * NCATS, activation='relu'))

#     model.add(BatchNormalization())

#     model.add(Dropout(0.4))

#     model.add(Dense(NCATS, activation='softmax'))



    model.summary()

    

    return model
def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
model = CNN_model()



model.compile(optimizer=Adam(lr=0.0024), loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
callbacks = [ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)]
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)

x_valid = df_to_image(valid_df)

y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)

print(x_valid.shape, y_valid.shape)

print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))
history = model.fit_generator(train_datagen, epochs = epochs, verbose = 1, 

                              validation_data=(x_valid, y_valid),

                              steps_per_epoch=x_valid.shape[0] // batch_size, callbacks=callbacks)
test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))

test.head()
x_test = df_to_image(test)

print(test.shape, x_test.shape)

print('Test array memory {:.2f} GB'.format(x_test.nbytes / 1024.**3 ))
test_predictions = model.predict(x_test, batch_size=batch_size)
top3 = pd.DataFrame(np.argsort(-test_predictions, axis=1)[:, :3])

top3.head()
word = top3.replace(classreverse_dict)

test['word'] = word[0] + ' ' + word[1] + ' ' + word[2]

submission = test[['key_id', 'word']]

submission.to_csv('submission.csv', index=False)

submission.head()
end = dt.datetime.now()

print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))