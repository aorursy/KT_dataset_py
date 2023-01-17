%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import os

import json

import datetime as dt

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 14

import seaborn as sns

import cv2

import pandas as pd

import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation

from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from tensorflow.keras.models import Sequential

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import MobileNet

from tensorflow.keras.applications.mobilenet import preprocess_input

start = dt.datetime.now()
DP_DIR = '../input/shuffle-csvs/'

INPUT_DIR = '../input/quickdraw-doodle-recognition/'



BASE_SIZE = 256

NCSVS = 100

NCATS = 340

np.random.seed(seed=1987)

tf.set_random_seed(seed=1987)



def f2cat(filename: str) -> str:

    return filename.split('.')[0]



def list_all_categories():

    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))

    return files
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



def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
STEPS = 800

EPOCHS = 9

size = 64

batchsize = 680
model = MobileNet(input_shape=(size, size, 3), alpha=1., weights=None, classes=NCATS)

model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',

              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])

print(model.summary())
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):

    img = np.zeros((BASE_SIZE, BASE_SIZE,3), np.uint8)

    

    colors = [ (255,0,0),

               (255,127,0),

               (255,255,0),

               (0,255,0),

               (0,0,255),

               (75,0,130),

               (139,0,255)]

    

    for t, stroke in enumerate(raw_strokes):

        color = ( 255 - colors[t % 7][0], 255 - colors[t % 7][1], 255 - colors[t % 7][2])

        for i in range(len(stroke[0]) - 1):

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

#     print(np.unique(img))

    if size != BASE_SIZE:

        return cv2.resize(img, (size, size))

    else:

        return img



def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):

    while True:

        for k in np.random.permutation(ks):

            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))

            for df in pd.read_csv(filename, chunksize=batchsize):

                df['drawing'] = df['drawing'].apply(json.loads)

                x = np.zeros((len(df), size, size, 3))

                for i, raw_strokes in enumerate(df.drawing.values):

                    x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw,

                                             time_color=time_color)

                x = preprocess_input(x).astype(np.float32)

                y = keras.utils.to_categorical(df.y, num_classes=NCATS)

                yield x, y



def df_to_image_array_xd(df, size, lw=6, time_color=True):

    df['drawing'] = df['drawing'].apply(json.loads)

    x = np.zeros((len(df), size, size, 3))

    for i, raw_strokes in enumerate(df.drawing.values):

#         print(raw_strokes)

        x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)

    x = preprocess_input(x).astype(np.float32)

    return x
valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)

# valid_df = pd.read_csv(os.path.join(INPUT_DIR, 'train_simplified'))

x_valid = df_to_image_array_xd(valid_df, size)

y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)

print(x_valid.shape, y_valid.shape)

print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))
train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))
x, y = next(train_datagen)

n = 8

fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))

for i in range(n**2):

    ax = axs[i // n, i % n]

    ax.imshow( (255*(-x[i, :, :, :] + 1)/2).astype(np.uint8) )

    ax.axis('off')

plt.tight_layout()

fig.savefig('gs.png', dpi=300)

plt.show();
%%timeit

x, y = next(train_datagen)
callbacks = [

    ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.75, patience=3, min_delta=0.001,

                          mode='max', min_lr=1e-5, verbose=1),

    ModelCheckpoint('model.h5', monitor='val_top_3_accuracy', mode='max', save_best_only=True,

                    save_weights_only=True),

]

hists = []

hist = model.fit_generator(

    train_datagen, steps_per_epoch=STEPS, epochs=1, verbose=1,

    validation_data=(x_valid, y_valid),

    callbacks = callbacks

)

hists.append(hist)
valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)

map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)

print('Map3: {:.3f}'.format(map3))
from tqdm import tqdm

from dask import bag
from PIL import Image, ImageDraw 

from pprint import pprint

import ast

imheight, imwidth = 64, 64 

# # faster conversion function



# for i, raw_strokes in enumerate(df.drawing.values):

#     x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)

    

def df_to_image_array_chunk(chunk_values, size=64, lw=6, time_color=True):

#     df['drawing'] = df['drawing'].apply(json.loads)

    x = np.zeros((len(chunk_values),size, size, 3), np.uint8)

    for i, raw_strokes in enumerate(chunk_values):

        x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)

    x = preprocess_input(x).astype(np.float32)

    return cv2.resize(x, (size, size))



def draw_it(raw_strokes, size=64, lw=6, time_color=True):

    img = np.zeros((BASE_SIZE, BASE_SIZE,3), np.uint8)

    colors = [ (255,0,0),

               (255,127,0),

               (255,255,0),

               (0,255,0),

               (0,0,255),

               (75,0,130),

               (139,0,255)]

    print(len(raw_strokes), len(raw_strokes[0]))

    for t, stroke in enumerate(raw_strokes):

#         if t == 1 or t == 2:

#             print(stroke)

        color = ( 255 - colors[t % 7][0], 255 - colors[t % 7][1], 255 - colors[t % 7][2])

        for i in range(len(stroke[0]) - 1):

            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),

                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)

#     pprint(np.unique(img))

    img = preprocess_input(img).astype(np.float32)

    return cv2.resize(img, (size, size))
ttvlist = []

reader = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'), index_col=['key_id'],

    chunksize=2048)

for df in tqdm(reader, total=55):

    df['drawing'] = df['drawing'].apply(json.loads)

    

    x = np.zeros((len(df), size, size, 3))

    for i, raw_strokes in enumerate(df.drawing.values):

        x[i, :, :, :] = draw_cv2(raw_strokes, size=size)

    x = preprocess_input(x).astype(np.float32)

    

    testpreds = model.predict(x, verbose=0)

    ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3

    ttvlist.append(ttvs)

    

ttvarray = np.concatenate(ttvlist)
# #%% set label dictionary and params

# classfiles = list_all_categories()

# numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)}
classfiles = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))

numstonames = {i: v[:-4].replace(" ", "_") for i, v in enumerate(classfiles)} #adds underscores
print(numstonames)
preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

preds_df = preds_df.replace(numstonames)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



sub = pd.read_csv('../input/quickdraw-doodle-recognition/sample_submission.csv', index_col=['key_id'])

sub['word'] = preds_df.words.values

# sub.to_csv('../input/quickdraw-doodle-recognition/submission_mobilenet.csv')

sub.to_csv('submission_mobilenet.csv')

sub.head()
preds_df.head()
# test = sub

# submission = test[['key_id', 'word']]

# submission.to_csv('gs_mn_submission_{}.csv'.format(int(map3 * 10**4)), index=False)

# submission.head()

# submission.shape
end = dt.datetime.now()

print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))