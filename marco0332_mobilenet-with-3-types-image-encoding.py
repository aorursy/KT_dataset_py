import numpy as np

import pandas as pd

import cv2



import json

import datetime as dt

from tqdm import tqdm



import ast

import math

from glob import glob

import glob

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from multiprocessing.dummy import Pool

from keras.models import load_model

import time

import keras

import random



from skimage.draw import draw

import matplotlib.pyplot as plt

import matplotlib.style as style

%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import os

print(os.listdir("../input/mobilenetfile"))

print(os.listdir("./"))
def f2cat(filename: str) -> str:

    return filename.split('.')[0]



class Simplified():

    def __init__(self, input_path='./input'):

        self.input_path = input_path



    def list_all_categories(self):

        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))

        return sorted([f2cat(f) for f in files], key=str.lower)



    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):

        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),

                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)

        if drawing_transform:

            df['drawing'] = df['drawing'].apply(json.loads)

        return df
# shuffle csv 만든적이 없다면 주석 풀고 실행 #



# PATH = '../input/quickdraw-doodle-recognition'



# start = dt.datetime.now()

# s = Simplified(PATH)

# NCSVS = 100

# categories = s.list_all_categories()

# print(len(categories))



# for y, cat in tqdm(enumerate(categories)):

#     df = s.read_training_csv(cat, nrows=30000)

#     df['y'] = y

#     df['cv'] = (df.key_id // 10 ** 7) % NCSVS

#     for k in range(NCSVS):

#         filename = 'train_k{}.csv'.format(k)

#         chunk = df[df.cv == k]

#         chunk = chunk.drop(['key_id'], axis=1)

#         if y == 0:

#             chunk.to_csv(filename, index=False)

#         else:

#             chunk.to_csv(filename, mode='a', header=False, index=False)



# for k in tqdm(range(NCSVS)):

#     filename = 'train_k{}.csv'.format(k)

#     if os.path.exists(filename):

#         df = pd.read_csv(filename)

#         df['rnd'] = np.random.rand(len(df))

#         df = df.sort_values(by='rnd').drop('rnd', axis=1)

#         df.to_csv(filename + '.gz', compression='gzip', index=False)

#         os.remove(filename)

# print(df.shape)



# end = dt.datetime.now()

# print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
INPUT_DIR = '../input/quickdraw-doodle-recognition/'

BASE_SIZE = 256



# Cross Validation을 위해 추가

def split_train_val(): 

    ALL_FILES = glob.glob('../input/shuffle-csvs/*.csv.gz')

    VALIDATION_FILE = '../input/shuffle-csvs/train_k'+str(int(random.random()*93))+'.csv.gz'

    ALL_FILES.remove(VALIDATION_FILE)

    np.random.seed(seed=1987)

    return ALL_FILES, VALIDATION_FILE





def apk(actual, predicted, k=3):

    """

    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    """

    if len(predicted) > k:

        predicted = predicted[:k]



    score = 0.0

    num_hits = 0.0



    for i, p in enumerate(predicted):

        if p in actual and p not in predicted[:i]:

            num_hits += 1.0

            score += num_hits / (i + 1.0)



    if not actual:

        return 0.0



    return score / min(len(actual), k)



def mapk(actual, predicted, k=3):

    """

    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py

    """

    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])





def preds2catids(predictions):

    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])



def f2cat(filename: str) -> str:

    return filename.split('.')[0]



def list_all_categories():

    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))

    return sorted([f2cat(f) for f in files], key=str.lower)





def plot_batch(x):    

    cols = 4

    rows = 6

    fig, axs = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True, figsize=(18, 18))

    for i in range(rows):

        for k in range(0,3):

            ax = axs[i, k]

            ax.imshow(x[i, :, :, k], cmap=plt.cm.gray)

            ax.axis('off')

        ax = axs[i, 3]

        ax.imshow(x[i, :, :], )

        ax.axis('off')

    fig.tight_layout()

    plt.show();
AUGMENTATION = True

STEPS = 200

BATCH_SIZE = 400

EPOCHS = 10

NCATS = 340

LEARNING_RATE = 0.002



IMG_SHAPE = (128,128,3)

IMG_SIZE = IMG_SHAPE[0]
def draw_cv2(raw_strokes, size=256, lw=6, augmentation = False):

    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)

    for t, stroke in enumerate(raw_strokes):

        points_count = len(stroke[0]) - 1

        grad = 255//points_count

        for i in range(len(stroke[0]) - 1):

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), (255, 255 - min(t,10)*13, max(255 - grad*i, 20)), lw)

    if size != BASE_SIZE:

        img = cv2.resize(img, (size, size))

    if augmentation:

        if random.random() > 0.5:

            img = np.fliplr(img)

    return img
def image_generator(size, batchsize, lw=6, augmentation = False):

    while True:

        for filename in ALL_FILES:

            for df in pd.read_csv(filename, chunksize=batchsize):

                df['drawing'] = df['drawing'].apply(eval)

                x = np.zeros((len(df), size, size,3))

                for i, raw_strokes in enumerate(df.drawing.values):

                    x[i] = draw_cv2(raw_strokes, size=size, lw=lw, augmentation = augmentation)

                x = x / 255.

                x = x.reshape((len(df), size, size, 3)).astype(np.float32)

                y = keras.utils.to_categorical(df.y, num_classes=NCATS)

                yield x, y



def valid_generator(valid_df, size, batchsize, lw=6):

    while(True):

        for i in range(0,len(valid_df),batchsize):

            chunk = valid_df[i:i+batchsize]

            x = np.zeros((len(chunk), size, size,3))

            for i, raw_strokes in enumerate(chunk.drawing.values):

                x[i] = draw_cv2(raw_strokes, size=size, lw=lw)

            x = x / 255.

            x = x.reshape((len(chunk), size, size,3)).astype(np.float32)

            y = keras.utils.to_categorical(chunk.y, num_classes=NCATS)

            yield x,y

        

def test_generator(test_df, size, batchsize, lw=6):

    for i in range(0,len(test_df),batchsize):

        chunk = test_df[i:i+batchsize]

        x = np.zeros((len(chunk), size, size,3))

        for i, raw_strokes in enumerate(chunk.drawing.values):

            x[i] = draw_cv2(raw_strokes, size=size, lw=lw)

        x = x / 255.

        x = x.reshape((len(chunk), size, size, 3)).astype(np.float32)

        yield x

        



ALL_FILES, VALIDATION_FILE = split_train_val()

train_datagen = image_generator(size=IMG_SIZE, batchsize=BATCH_SIZE, augmentation = AUGMENTATION)



valid_df = pd.read_csv(VALIDATION_FILE)

valid_df['drawing'] = valid_df['drawing'].apply(eval)

validation_steps = len(valid_df)//BATCH_SIZE

valid_datagen = valid_generator(valid_df, size=IMG_SIZE, batchsize=BATCH_SIZE)
single_class_df = valid_df[valid_df['y'] == 2]

single_class_gen = valid_generator(single_class_df, size=IMG_SIZE, batchsize=BATCH_SIZE)

x, y = next(single_class_gen)

plot_batch(x)
from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Activation

from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from keras.models import Sequential

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.optimizers import Adam

from keras.applications.mobilenet import MobileNet

from keras.applications.mobilenet import preprocess_input

from keras.models import load_model



def top_3_accuracy(y_true, y_pred):

    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)



reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

checkpointer = ModelCheckpoint(filepath='mobileNet_ckpt.hdf5', verbose=2, save_best_only=True)

model = load_model('../input/mobilenetfile/mobileNet.hdf5', custom_objects = {'top_3_accuracy':top_3_accuracy})

opt = Adam(lr = LEARNING_RATE)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy', top_3_accuracy])

model.summary()
history = model.fit_generator(train_datagen,

                              steps_per_epoch=STEPS,

                              epochs=EPOCHS,

                              verbose=2,

                              validation_data=valid_datagen,

                              validation_steps=validation_steps,

                              callbacks=[checkpointer,reducer])

model.save('mobileNet.hdf5')
submission_df = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))

submission_df['drawing'] = submission_df['drawing'].apply(eval)

submission_datagen = test_generator(submission_df, size=IMG_SIZE, batchsize=BATCH_SIZE)

submission_predictions = model.predict_generator(submission_datagen, math.ceil(len(submission_df)/BATCH_SIZE))

cats = list_all_categories()

id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}

top3 = preds2catids(submission_predictions)

top3cats = top3.replace(id2cat)

submission_df['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']

submission = submission_df[['key_id', 'word']]

submission.to_csv('submission.csv', index=False)