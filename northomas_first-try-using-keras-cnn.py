# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import os

import math

import cv2



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from mpl_toolkits.axes_grid1 import ImageGrid



from PIL import Image



import seaborn as sns



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Lambda

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2

from keras.callbacks import ModelCheckpoint



from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



%matplotlib inline
filenames = os.listdir('../input/train-jpg')

df = pd.read_csv('../input/train.csv')
df.info()
df.describe()
df['tag_set'] = df['tags'].map(lambda s: set(s.split(' ')))



tags = set()

for t in df['tags']:

    s = set(t.split(' '))

    tags = tags | s



tag_list = list(tags)

tag_list.sort()

tag_columns = ['tag_' + t for t in tag_list]

for t in tag_list:

    df['tag_' + t] = df['tag_set'].map(lambda x: 1 if t in x else 0)
df.info()

df.describe()
df.head()
df[tag_columns].sum()
df[tag_columns].sum().sort_values().plot.bar()
tags_count = df.groupby('tags').count().sort_values(by='image_name', ascending=False)['image_name']

print('There are {} unique tag combinations'.format(len(tags_count)))

print()

print(tags_count)
from textwrap import wrap



def display(images, cols=None, maxcols=10, width=14, titles=None):

    if cols is None:

        cols = len(images)

    n_cols = cols if cols < maxcols else maxcols

    plt.rc('axes', grid=False)

    fig1 = plt.figure(1, (width, width * math.ceil(len(images)/n_cols)))

    grid1 = ImageGrid(

                fig1,

                111,

                nrows_ncols=(math.ceil(len(images)/n_cols), n_cols),

                axes_pad=(0.1, 0.6)

            )



    for index, img in enumerate(images):

        grid1[index].grid = False

        if titles is not None:

            grid1[index].set_title('\n'.join(wrap(titles[index], width=25)))

        if len(img.shape) == 2:

            grid1[index].imshow(img, cmap='gray')

        else:

            grid1[index].imshow(img)
def load_image(filename, resize=True, folder='train-jpg'):

    img = mpimg.imread('../input/{}/{}.jpg'.format(folder, filename))

    if resize:

        img = cv2.resize(img, (64, 64))

    return np.array(img)



def mean_normalize(img):

    return (img - img.mean()) / (img.max() - img.min())



def normalize(img):

    return img / 127.5 - 1
samples = df.sample(16)

sample_images = [load_image(fn) for fn in samples['image_name']]

INPUT_SHAPE = sample_images[0].shape

print(INPUT_SHAPE)

display(

    sample_images,

    cols=4,

    titles=[t for t in samples['tags']]

)
def preprocess(img):

    img = normalize(img)

    return img



display(

    [(127.5 * (preprocess(img) + 1)).astype(np.uint8) for img in sample_images],

    cols=4,

    titles=[t for t in samples['tags']]

)
df_train = df
X = df_train['image_name'].values

y = df_train[tag_columns].values



n_features = 1

n_classes = y.shape[1]



X, y = shuffle(X, y)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)



print('We\'ve got {} feature rows and {} labels'.format(len(X_train), len(y_train)))

print('Each row has {} features'.format(n_features))

print('and we have {} classes'.format(n_classes))

assert(len(y_train) == len(X_train))

print('We use {} rows for training and {} rows for validation'.format(len(X_train), len(X_valid)))

print('Each image has the shape:', INPUT_SHAPE)

print('So far, so good')
print('Memory usage (train) kB', X_train.nbytes//(1024))

print('Memory usage (valid) kB', X_valid.nbytes//(1024))
def generator(X, y, batch_size=32):

    X_copy, y_copy = X, y

    while True:

        for i in range(0, len(X_copy), batch_size):

            X_result, y_result = [], []

            for x, y in zip(X_copy[i:i+batch_size], y_copy[i:i+batch_size]):

                rx, ry = [load_image(x)], [y]

                rx = np.array([preprocess(x) for x in rx])

                ry = np.array(ry)

                X_result.append(rx)

                y_result.append(ry)

            X_result, y_result = np.concatenate(X_result), np.concatenate(y_result)

            yield shuffle(X_result, y_result)

        X_copy, y_copy = shuffle(X_copy, y_copy)
from keras import backend as K



def fbeta(y_true, y_pred, threshold_shift=0):

    beta = 2



    # just in case of hipster activation at the final layer

    y_pred = K.clip(y_pred, 0, 1)



    # shifting the prediction threshold from .5 if needed

    y_pred_bin = K.round(y_pred + threshold_shift)



    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()

    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))



    precision = tp / (tp + fp)

    recall = tp / (tp + fn)



    beta_squared = beta ** 2

    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())



# ---------------------------------- #



model = Sequential()



model.add(Conv2D(48, (8, 8), strides=(2, 2), input_shape=INPUT_SHAPE, activation='elu'))

model.add(BatchNormalization())



model.add(Conv2D(64, (8, 8), strides=(2, 2), activation='elu'))

model.add(BatchNormalization())



model.add(Conv2D(96, (5, 5), strides=(2, 2), activation='elu'))

model.add(BatchNormalization())



model.add(Conv2D(96, (3, 3), activation='elu'))

model.add(BatchNormalization())



model.add(Flatten())

model.add(Dropout(0.3))



model.add(Dense(256, activation='elu'))

model.add(BatchNormalization())



model.add(Dense(64, activation='elu'))

model.add(BatchNormalization())



model.add(Dense(n_classes, activation='sigmoid'))



    

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[fbeta, 'accuracy']

)



model.summary()
EPOCHS = 6

BATCH = 32

PER_EPOCH = 256



X_train, y_train = shuffle(X_train, y_train)

X_valid, y_valid = shuffle(X_valid, y_valid)



filepath="weights-improvement-{epoch:02d}-{val_fbeta:.3f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_fbeta', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]



history = model.fit_generator(

    generator(X_train, y_train, batch_size=BATCH),

    steps_per_epoch=PER_EPOCH,

    epochs=EPOCHS,

    validation_data=generator(X_valid, y_valid, batch_size=BATCH),

    validation_steps=len(y_valid)//(4*BATCH),

    callbacks=callbacks_list

)
X_test = os.listdir('../input/test-jpg')

X_test = [fn.replace('.jpg', '') for fn in X_test]



result = []

TEST_BATCH = 128

for i in range(0, len(X_test), TEST_BATCH):

    X_batch = X_test[i:i+TEST_BATCH]

    X_batch = np.array([preprocess(load_image(fn, folder='test-jpg')) for fn in X_batch])

    p = model.predict(X_batch)

    result.append(p)

    

r = np.concatenate(result)

r = r > 0.5



table = []

for row in r:

    t = []

    for b, v in zip(row, tag_columns):

        if b:

            t.append(v.replace('tag_', ''))

    table.append(' '.join(t))



df_pred = pd.DataFrame.from_dict({'image_name': X_test, 'tags': table})

df_pred.to_csv('submission.csv', index=False)