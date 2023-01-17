%matplotlib inline

import os

import cv2

import json

import pandas as pd

import numpy as np

from glob import glob

from keras.utils.np_utils import to_categorical
SHUFFLE_DIR = 'E:\\project\\shuffle_data'

TEST_DIR = 'E:\\project\\data\\'
base_size = 256

shuffle_data_num = 100

num_classes = 340

img_size = 64

epochs = 20

steps = 600

batch_size = 800
TRAIN_LIST = glob('E:\\project\\data\\train_simplified\\*.csv')

class_list = []
for item in TRAIN_LIST:

    class_name = os.path.basename(item).split('.')[0]

    class_name = class_name.replace(' ', '_')

    class_list.append(class_name)
class_list[:5]
valid_df = pd.read_csv(os.path.join(SHUFFLE_DIR, 'train_k99.csv.gz'), nrows=34000)
valid_df.head()
def drawing(raw_strokes, img_size, lw=6, time_color=True):

    img = np.zeros((256, 256), np.uint8)

    for t, stroke in enumerate(raw_strokes):

        color = 255 - min(t, 10) * 13

        for i in range(len(stroke[0]) - 1):

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

    if img_size != 256:

        return cv2.resize(img, (img_size, img_size))/255

    else:

        return img/255
def data_generator(df, img_size, lw=6, time_color=True):

    df['drawing'] = df['drawing'].apply(json.loads)

    x = np.zeros((len(df), img_size, img_size, 1))

    for i, raw_strokes in enumerate(df.drawing.values):

        x[i, :, :, 0] = drawing(raw_strokes, img_size=img_size, lw=lw, time_color=time_color)

    return x
x_valid = data_generator(valid_df, img_size)

y_valid = to_categorical(valid_df.y, num_classes=num_classes)
print(x_valid.shape, y_valid.shape)
def train_generator(img_size, batch_size, iters, lw=6, time_color=True):

    while True:

        for iter in np.random.permutation(iters):

            filename = os.path.join(SHUFFLE_DIR, 'train_k{}.csv.gz'.format(iter))

            for df in pd.read_csv(filename, chunksize=batch_size):

                df['drawing'] = df['drawing'].apply(json.loads)

                x = np.zeros((len(df), img_size, img_size, 1))

                for i, raw_strokes in enumerate(df.drawing.values):

                    x[i, :, :, 0] = drawing(raw_strokes, img_size=img_size, lw=lw, time_color=time_color)

                y = to_categorical(df.y, num_classes=num_classes)

                yield x, y
train_data = train_generator(img_size, batch_size, range(shuffle_data_num-1))
x, y = next(train_data)
print(x.shape, y.shape)
import keras

from time import time

from keras import Model

from keras.models import load_model

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten

from keras.applications import MobileNet

from keras import optimizers

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
model = MobileNet(input_shape=(img_size, img_size, 1), alpha=1, weights=None, classes=340)
model.summary()
c = keras.optimizers.adam(lr=0.002)
model.compile(loss='categorical_crossentropy', optimizer=c, metrics=['accuracy', top_3_accuracy])
callbacks = [

    ReduceLROnPlateau(monitor='top_3_accuracy', factor=0.8, patience=3, min_delta=0.001,

                          mode='max', min_lr=1e-5, verbose=1),

    EarlyStopping(patience=3, monitor='top_3_accuracy'),

    ModelCheckpoint('model_mobilenet.h5', monitor='top_3_accuracy', mode='max', save_best_only=True,

                    save_weights_only=True),

    TensorBoard(log_dir="logs/{}".format(time()))

]
hist = model.fit_generator(

    train_data, steps_per_epoch=steps, epochs=epochs, verbose=1,

    validation_data=(x_valid, y_valid), callbacks=callbacks)
valid_preds = model.predict(x_valid, batch_size=batch_size, verbose=1)
answer = []

for i in range(len(valid_preds)):

    top3 = valid_preds[i].argsort()[::-1][:3]

    word = ''

    for j in top3:

        word += class_list[j]

        word += " "

    answer.append(word)

valid_preds_df = pd.DataFrame(answer)
valid_preds_df.head(20)
valid_df
test = pd.read_csv('E:\\project\\data\\test_simplified.csv')
test.head()
x_test = data_generator(test, img_size)
test_preds = model.predict(x_test, batch_size=batch_size, verbose=1)
def create_submission(test, test_preds):

    pred_rows = []

    answer = []

    for i in range(len(test_preds)):

        top3 = test_preds[i].argsort()[::-1][:3]

        word = ''

        for j in top3:

            word += class_list[j]

            word += " "

        answer.append(word)

    df = pd.DataFrame(answer)

    test['word'] = df

    sub = test[['key_id', 'word']]

    sub.to_csv('submission_{}.csv'.format(time()), index=False)
create_submission(test, test_preds)