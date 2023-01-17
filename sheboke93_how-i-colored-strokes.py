%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
import ast
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
from tensorflow.keras.applications.mobilenet import preprocess_input
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
    return sorted([f2cat(f) for f in files], key=str.lower)
size = 256
batchsize = 512
def draw_cv2_strokes1(raw_strokes, size=256, lw=4, index1=0, index2=1):
    img = np.zeros((size, size, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes[index1:index2]):
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i]*size//BASE_SIZE, stroke[1][i]*size//BASE_SIZE),
                         (stroke[0][i + 1]*size//BASE_SIZE, stroke[1][i + 1]*size//BASE_SIZE), (255,0,0), lw)
        return img


def draw_cv2_strokes3(raw_strokes, size=256, lw=4, index1=0, index2=1):
    img = np.zeros((size, size, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes[index1:index2]):
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i]*size//BASE_SIZE, stroke[1][i]*size//BASE_SIZE),
                         (stroke[0][i + 1]*size//BASE_SIZE, stroke[1][i + 1]*size//BASE_SIZE), (0,0,255), lw)
        return img


def draw_cv2_strokes2(raw_strokes, size=256, lw=4, index1=0, index2=1):
    img = np.zeros((size, size, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes[index1:index2]):
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i]*size//BASE_SIZE, stroke[1][i]*size//BASE_SIZE),
                         (stroke[0][i + 1]*size//BASE_SIZE, stroke[1][i + 1]*size//BASE_SIZE), (0,255,0), lw)
        return img


def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 3))
                for i, raw_strokes in enumerate(df.drawing.values):
                    num_of_strokes = len(raw_strokes)
                    if num_of_strokes == 1:
                        x[i, :, :, 0] = draw_cv2_strokes1(raw_strokes, size=size, lw=lw,
                                                          index1=0, index2=1)[:,:,0]
                        x[i, :, :, 1] = draw_cv2_strokes1(raw_strokes, size=size, lw=lw,
                                                          index1=0, index2=1)[:,:,1]
                        x[i, :, :, 2] = draw_cv2_strokes1(raw_strokes, size=size, lw=lw,
                                                          index1=0, index2=1)[:,:,2]
                    elif num_of_strokes == 2:
                        x[i, :, :, 0] = draw_cv2_strokes1(raw_strokes, size=size, lw=lw,
                                                          index1=0, index2=1)[:,:,0]
                        x[i, :, :, 1] = draw_cv2_strokes2(raw_strokes, size=size, lw=lw,
                                                          index1=1, index2=2)[:,:,1]
                        x[i, :, :, 2] = draw_cv2_strokes2(raw_strokes, size=size, lw=lw,
                                                         index1=0, index2=2)[:,:,2]
                    else:
                        x[i, :, :, 0] = draw_cv2_strokes1(raw_strokes, size=size, lw=lw,
                                                         index1=0, index2=num_of_strokes // 3)[:,:,0]
                        x[i, :, :, 1] = draw_cv2_strokes2(raw_strokes, size=size, lw=lw,
                                                          index1=num_of_strokes // 3,
                                                         index2=2 * num_of_strokes // 3)[:,:,1]
                        x[i, :, :, 2] = draw_cv2_strokes3(raw_strokes, size=size, lw=lw,
                                                         index1=2*num_of_strokes // 3,
                                                         index2=3 * num_of_strokes // 3)[:,:,2]

                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y
train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))
x, y = next(train_datagen)
n = 6
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x[i]+1)/2
    ax.imshow((x[i, :, :, :]+1)/2)
    ax.axis('off')
plt.tight_layout()
fig.savefig('gs.png', dpi=300)
plt.show();
color = [(187,255,255), (106, 90, 205),(0, 255, 127),(255,255,0),(255,193,37),(205,92,92),
         (244, 164, 96), (255, 105, 180),(218, 112, 214),(255, 165, 0),(139, 134, 130),(24, 116, 205),
         (187,255,255), (106, 90, 205),(0, 255, 127),(255,255,0),(255,193,37),(205,92,92),
         (244, 164, 96), (255, 105, 180),(218, 112, 214),(255, 165, 0),(139, 134, 130),(24, 116, 205),
        (187,255,255), (106, 90, 205),(0, 255, 127),(255,255,0),(255,193,37),(205,92,92),
         (244, 164, 96), (255, 105, 180),(218, 112, 214),(255, 165, 0),(139, 134, 130),(24, 116, 205)]

def draw_cv2_strokes(raw_strokes, size=256, lw=4, color = color):
    img = np.zeros((size, size, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        cl = color[t]
        for i in range(len(stroke[0]) - 1):
            _ = cv2.line(img, (stroke[0][i] * size // BASE_SIZE, stroke[1][i] * size // BASE_SIZE),
                         (stroke[0][i + 1] * size // BASE_SIZE, stroke[1][i + 1] * size // BASE_SIZE), cl, lw)
    return img

def image_generator_xd(size, batchsize, ks, lw=6):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                x = np.zeros((len(df), size, size, 3))
                for i, raw_strokes in enumerate(df.drawing.values):
                    num_of_strokes = len(raw_strokes)
                    if num_of_strokes == 1:
                        x[i, :, :, 0] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                           )[:, :, 0]
                        x[i, :, :, 1] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 1]
                        x[i, :, :, 2] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                           )[:, :, 2]
                    elif num_of_strokes == 2:
                        x[i, :, :, 0] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 0]
                        x[i, :, :, 1] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                           )[:, :, 1]
                        x[i, :, :, 2] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 2]
                    else:
                        x[i, :, :, 0] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 0]
                        x[i, :, :, 1] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                        )[:, :, 1]
                        x[i, :, :, 2] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 2]

                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y


def df_to_image_array_xd(df, size, lw=6):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size, 3))
    for i, raw_strokes in enumerate(df.drawing.values):
        num_of_strokes = len(raw_strokes)
        if num_of_strokes == 1:
            x[i, :, :, 0] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                           )[:, :, 0]
            x[i, :, :, 1] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 1]
            x[i, :, :, 2] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                           )[:, :, 2]
        elif num_of_strokes == 2:
            x[i, :, :, 0] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 0]
            x[i, :, :, 1] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                           )[:, :, 1]
            x[i, :, :, 2] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 2]
        else:
            x[i, :, :, 0] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 0]
            x[i, :, :, 1] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                        )[:, :, 1]
            x[i, :, :, 2] = draw_cv2_strokes(raw_strokes, size=size, lw=lw
                                                          )[:, :, 2]


    x = preprocess_input(x).astype(np.float32)
    return x
train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))
x, y = next(train_datagen)
n = 6
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    (-x[i]+1)/2
    ax.imshow((x[i, :, :, :]+1)/2)
    ax.axis('off')
plt.tight_layout()
fig.savefig('gs.png', dpi=300)
plt.show();
