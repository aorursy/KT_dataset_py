%matplotlib inline
import pandas as pd
import numpy as np
import os
import ast
import cv2

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14

from tensorflow.keras.applications.resnet50 import preprocess_input
BASE_SIZE = 256

img_size = 64
batchsize = 128
line_width = 7
colors = [(255, 0, 0) , (255, 255, 0),  (128, 255, 0),  (0, 255, 0), (0, 255, 128), (0, 255, 255), 
          (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255)]
def draw_cv2(raw_strokes, size=256, lw=7, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = colors[min(t, len(colors)-1)]
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw, lineType=cv2.LINE_AA)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img
def test_generator(img_size, batchsize, lw=6):
    while True:
        for df in pd.read_csv('../input/test_simplified.csv', chunksize=batchsize):
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
            x = np.zeros((len(df), img_size, img_size, 3))
            for i, raw_strokes in enumerate(df.drawing.values):
                x[i, :, :, :] = draw_cv2(raw_strokes, size=img_size, lw=lw)
            yield x, preprocess_input(x).astype(np.float32)
test_datagen = test_generator(img_size, batchsize, line_width)
x, xi = next(test_datagen)
n = 8
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    ax.imshow(x[i])
    ax.axis('off')
plt.tight_layout()
fig.savefig('gs.png', dpi=300)
plt.show();
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
for i in range(n**2):
    ax = axs[i // n, i % n]
    ax.imshow(xi[i]) # use ax.imshow(xi[i] * 255)
    ax.axis('off')
plt.tight_layout()
fig.savefig('gs.png', dpi=300)
plt.show();