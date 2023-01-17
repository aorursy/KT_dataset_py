import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import os
# import multiprocessing
import cv2
import math

from keras.applications import MobileNet
from keras.losses import sparse_categorical_crossentropy
from tqdm import trange
plt.rcParams["figure.max_open_warning"] = 300
def strokes_to_img(in_strokes):
    in_strokes = eval(in_strokes)
    # make an agg figure
    fig, ax = plt.subplots()
    for x,y in in_strokes:
        ax.plot(x, y, linewidth=12.) #  marker='.',
    ax.axis('off')
    fig.canvas.draw()
    
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    return (cv2.resize(X, (96, 96)) / 255.)[::-1]

class_files = os.listdir("../input/train_simplified/")
classes = {x[:-4]:i for i, x in enumerate(class_files)}
to_class = {i:x[:-4].replace(" ", "_") for i, x in enumerate(class_files)}
dfs = [pd.read_csv("../input/train_simplified/" + x, nrows=10000)[["word", "drawing"]] for x in class_files]
df = pd.concat(dfs)
del dfs
# mppool = multiprocessing.Pool(6)
n_samples = df.shape[0]
batch_size = 64

pick_order = np.arange(n_samples)
pick_per_epoch = n_samples // batch_size

def train_gen():
    while True:  # Infinity loop
        np.random.shuffle(pick_order)
        for i in range(pick_per_epoch):
            c_pick = pick_order[i*batch_size: (i+1)*batch_size]
            dfs = df.iloc[c_pick]
            out_imgs = list(map(strokes_to_img, dfs["drawing"]))
            X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
            y = np.array([classes[x] for x in dfs["word"]])
            yield X, y
tran_datagen = train_gen()
x,y = next(tran_datagen)
# Display some images
for i in range(12):
    plt.subplot(2,6,i+1)
    plt.imshow(x[i])
    plt.axis('off')
plt.show()
model = MobileNet(input_shape=(96, 96, 3), weights=None, classes=len(classes))
model.compile(optimizer="adam", loss=sparse_categorical_crossentropy)
model.fit_generator(tran_datagen, steps_per_epoch=20, epochs=5, verbose=1)

del tran_datagen
del df

import gc
gc.collect()
test_df = pd.read_csv("../input/test_simplified.csv")
n_samples = test_df.shape[0]
pick_per_epoch = math.ceil(n_samples / batch_size)
pick_order = np.arange(test_df.shape[0])

all_preds = []

for i in trange(pick_per_epoch):
        c_pick = pick_order[i*batch_size: (i+1)*batch_size]
        dfs = test_df.iloc[c_pick]
        out_imgs = list(map(strokes_to_img, dfs["drawing"]))
        X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
        preds = model.predict(X)
        for x in preds:
            all_preds.append(to_class[np.argmax(x)])
        if i == 50:  # TODO: let it run till completion
            break
fdf = pd.DataFrame({"key_id": test_df["key_id"], "word": all_preds + ([""] * (test_df.shape[0] - len(all_preds)))})  # TODO: No need to kill it early
fdf.to_csv("mobilenet_submit.csv", index=False)