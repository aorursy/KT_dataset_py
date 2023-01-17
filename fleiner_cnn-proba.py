import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import numpy as np # linear algebra
import pandas as pd
from os import listdir
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2
csv_train = pd.read_csv("/kaggle/input/football-player-number-13/train_solutions.csv")
csv_test = pd.read_csv("/kaggle/input/football-player-number-13/sampleSubmissionAllZeros.csv")
ids, labels = csv_train["Id"].to_numpy(), csv_train["Predicted"].to_numpy()
x_train=[]
y_train=[]
for i in ids[labels]:
    pos=i.find("-")
    num=int(i[pos+1:])
    trueid=i[:pos]+"_"+str(num)
    im = cv2.imread("/kaggle/input/football-player-number-13/images/" + trueid + '.jpg', cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#    im = im / 255.0
    im = im.astype('float32')
    x_train.append(im)
    y_train.append(1)
    if num > 2:
        falseid1=i[:pos]+"_"+str(num-2)
        im = cv2.imread("/kaggle/input/football-player-number-13/images/" + falseid1 + '.jpg', cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = im.astype('float32')
        x_train.append(im)
        y_train.append(0)
    if num <15:
        falseid2=i[:pos]+"_"+str(num+2)
        im = cv2.imread("/kaggle/input/football-player-number-13/images/" + falseid2 + '.jpg', cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = im.astype('float32')
        x_train.append(im)
        y_train.append(0)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = []
for i in csv_test["Id"].to_numpy():
    pos=i.find("-")
    num=int(i[pos+1:])
    trueid=i[:pos]+"_"+str(num)
    im = cv2.imread("/kaggle/input/football-player-number-13/images/" + trueid + '.jpg', cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#    im = im / 255.0
    im = im.astype('float32')
    x_test.append(im)
x_test = np.array(x_test) 
x_train = [resize(img, (280, 420)) for img in x_train]
x_test = [resize(img, (280, 420)) for img in x_test]
x_train = np.array(x_train)
#y_train = np.array(y_train)
x_test = np.array(x_test)
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPool2D
model = Sequential()
model.add(Conv2D(40, kernel_size=(3,3), padding='same', activation='relu', input_shape=(280, 420, 3)))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(20, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(10, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(300, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(x_train, y_train, validation_split= 0.33, epochs= 5)
predictions = model.predict(x_test)

pred = [1 if x<0.9 else 0 for x in predictions]

x=0
for i in pred:
    if i>0:
        x+=1
print(x)
csv_test["Predicted"]= pred
csv_test.to_csv("s.csv", index=False)
