# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir("../input/germantrafficsigndataset/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train"))

from PIL import Image

import random

import matplotlib.pyplot as plot

# Any results you write to the current directory are saved as output.
def GenerateDF(path, dummy_classes=['Pedestrian', 'Parking', 'SpeedLimit', 'DoNotEnter', 'GiveWay', 'Stop', 'TurnRight']):

    classes = dummy_classes

    

    class_list = []

    path_list = []

    for c in classes:

        for file in os.listdir(os.path.join(path, c)):

            total_path = os.path.join(os.path.join(path, c), file)

            path_list.append(total_path)

            class_list.append(c)

            

    class_list = pd.Series(class_list)

    class_list = pd.get_dummies(class_list, columns=dummy_classes)

    path_list = pd.Series(path_list).rename("path")

    

    return pd.concat([path_list, class_list], 1)
train_data = GenerateDF("../input/germantrafficsigndataset/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/train").sample(frac=1).reset_index().drop("index", 1)

print(train_data.shape)

train_data.head()
val_data = train_data.sample(frac=0.2)

train_data = train_data.drop(val_data.index)

print(train_data.shape)
test_data = GenerateDF("../input/germantrafficsigndataset/GermanTrafficSignDatasetWithBackdoorImages/CleanDataset/test").sample(frac=1).reset_index().drop("index", 1)

print(test_data.shape)

print(test_data.iloc[0].path)

print(test_data.iloc[0])



test_data.head()
def LoadImage(path, augmentation=False):

    img = Image.open(path).resize((35,35))

    

    if augmentation:

        angle = (random.random()*2-1)*15.0

        img = img.rotate(angle)

    img = np.array(img)

    img = img-np.min(img)

    img = img/np.max(img)

    

    return img



img = LoadImage(train_data.sample(1).values[0][0])

print(img.shape)

plot.imshow(img)
def Generate(data, batch_size=15):

    while True:

        batch = data.sample(batch_size)

        imgs = []

        classes = []

        for i, row in batch.iterrows():

            try:

                img = LoadImage(row["path"]).astype("float")

                c = row.drop("path").values.astype("float")



                imgs.append(img)

                classes.append(c)

            except Exception as e:

                print(e)

        yield np.array(imgs), np.array(classes)

        

gen = Generate(train_data)

imgs, sols = next(gen)

print(imgs.shape)

print(sols.shape)

print(sols)
from keras.models import Model, load_model

from keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, LeakyReLU, Dropout

from keras.optimizers import SGD, Adam

from keras.callbacks import ModelCheckpoint



inp = Input(shape=(35,35,3))

x = BatchNormalization()(inp)

x = Conv2D(256, 5, activation="relu")(x)

x = MaxPooling2D()(x)

x = LeakyReLU()(x)

x = Conv2D(128, 3, activation="relu")(x)

x = MaxPooling2D()(x)

x = Conv2D(64, 3, activation="relu")(x)

x = Flatten()(x)

x = Dropout(0.5)(x)

x = Dense(7, activation="softmax")(x)







model = Model(inp, x)

model.compile(optimizer=SGD(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])



model.summary()
from keras.utils import plot_model

plot_model(model, to_file='model.png')
train_gen = Generate(train_data)

val_gen = Generate(val_data)
clb = [ModelCheckpoint("best.h5", save_best_only=True, verbose=0)]

start = time.time()

h = model.fit_generator(train_gen, epochs=200, steps_per_epoch=10, validation_data=val_gen, validation_steps=5, verbose=1, callbacks=clb)
print("Training time: "+str(time.time()-start))
plot.figure(figsize=(8,6))

plot.plot(h.history["loss"], label="loss")

plot.plot(h.history["val_loss"], label="val. loss")

plot.xlabel("Epoch")

plot.ylabel("Loss")

plot.tight_layout()

plot.legend()

plot.savefig("loss.png")

plot.show()
imgs = []

classes = []

for i, row in test_data.iterrows():

    img = LoadImage(row["path"]).astype("float")

    c = row.drop("path").values.astype("float")

    print(c)



    imgs.append(img)

    classes.append(c)



imgs = np.array(imgs)

classes = np.array(classes)

print(imgs.shape)

print(classes.shape)

print(np.sum(classes,0))
start = time.time()

p = model.predict(imgs)

print("Prediction time: "+str(time.time()-start))

p = np.argmax(p, axis=1)



Y_true = np.argmax(classes, axis=1)

print(Y_true)

cols = test_data.columns[1:]
from sklearn.metrics import confusion_matrix

import seaborn as sns

M = confusion_matrix(Y_true, p)

print(M)



plot.figure(figsize=(8,6))

sns.heatmap(M, annot=True, fmt='g')

plot.xticks(np.arange(M.shape[0]), cols, rotation=90)

plot.yticks(np.arange(M.shape[0]), cols, rotation=0)

plot.xlabel("Predicted")

plot.ylabel("True")

plot.savefig("confusion.png")

plot.show()
from sklearn.metrics import accuracy_score as acc



print(acc(Y_true, p))