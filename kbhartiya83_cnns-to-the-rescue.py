# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from glob import glob
import cv2
#import tensorflow as tf
#from tensorflow.keras.metrics import top_k_categorical_accuracy
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from keras.applications import MobileNet
from keras.losses import categorical_crossentropy
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
files = glob('../input/train_simplified/*.csv')
col_name = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
draw_list = []
for f in files:
    df = pd.read_csv(f,nrows=10000)
    df = df[df.recognized==True]
    draw_list.append(df)
drawing_df = pd.DataFrame(np.concatenate(draw_list),columns=col_name)

drawing_df = drawing_df[["word","drawing"]]
del df
del draw_list
drawing_df.head()
    
def strokes_to_img(strokes):
    strokes = eval(strokes)
    fig, ax = plt.subplots()
    for x, y in strokes:
        ax.plot(x,y,linewidth=12.)
    ax.axis('off')
    fig.canvas.draw()
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    return (cv2.resize(X, (96, 96)) / 255.)[::-1]
    
class_files = os.listdir("../input/train_simplified/")
classes_to_idx = {x.split('.')[0]:i for i, x in enumerate(class_files)}
idx_to_classes = {i:x.split('.')[0].replace(" ","_") for i, x in enumerate(class_files)}
n_samples = drawing_df.shape[0]
batch_size = 64

pick_order = np.arange(n_samples)
pick_per_epoch = n_samples // batch_size

def train_gen():
    while True:  # Infinity loop
        np.random.shuffle(pick_order)
        for i in range(pick_per_epoch):
            c_pick = pick_order[i*batch_size: (i+1)*batch_size]
            dfs = drawing_df.iloc[c_pick]
            out_imgs = list(map(strokes_to_img, dfs["drawing"]))
            X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
            y = np.array([classes_to_idx[x] for x in dfs["word"]])
            yield X, y
train_datagen = train_gen()
x,y = next(train_datagen)
model = MobileNet(input_shape=(96, 96, 3), weights=None, classes=len(classes_to_idx))
model.summary()


model.compile(optimizer="adam", loss=categorical_crossentropy,metrics=['accuracy'])
model.fit_generator(train_datagen, steps_per_epoch=20, epochs=8, verbose=1)
test = pd.read_csv("../input/test_simplified.csv")

test_samples = test.shape[0]
pick_order = np.arrange(test_samples)
pick_per_epoch = test_samples // batch_size
all_preds = []

for i in trange(pick_per_epoch):
        c_pick = pick_order[i*batch_size: (i+1)*batch_size]
        dfs = test.iloc[c_pick]
        out_imgs = list(map(strokes_to_img, dfs["drawing"]))
        X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
        preds = model.predict(X)
        for x in preds:
            all_preds.append(idx_to_class[np.argmax(x)])

sdf = pd.DataFrame({"key_id": test["key_id"], "word": all_preds + ([""] * (test.shape[0] - len(all_preds)))})
sdf.to_csv("first_submit.csv", index=False)