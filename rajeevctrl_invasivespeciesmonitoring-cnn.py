# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def shuffle(lst):
    arr = lst[0]
    indicies = np.random.permutation(np.arange(len(arr)))
    result = []
    for l in lst:
        l = np.array(l)[indicies]
        result.append(l)
    return result

def readImage(imgPath,outputShape):
    frame = cv2.imread(imgPath,cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame,(outputShape[0],outputShape[1]))
    return frame
home = "../input"
trainHome = home+"/train"
testHome = home+"/test"
trainLabels = pd.read_csv(home+"/train_labels.csv")

outputImgShape = (200,200,3)
img_to_class = dict((str(row[0])+".jpg",str(row[1]))for row in trainLabels.as_matrix())
#img_to_class = shuffle([list(img_to_class.items())])[0]
#img_to_class = dict((str(row[0]),int(row[1])) for row in img_to_class)
imgNames = os.listdir(trainHome)
imgNames = shuffle([imgNames])[0].tolist()

#valLen = int(len(imgNames)*0.2)
#valImgNames = imgNames[:valLen]
#trainImgNames = imgNames[valLen:]

print("Total imgs train home",len(imgNames))
#print("Train images",len(trainImgNames))
#print("Val images",len(valImgNames))

X = []
Y = []

for imgName in tqdm(imgNames):
    imgClass = int(img_to_class[imgName])
    img = readImage(trainHome+"/"+imgName,outputImgShape)
    X.append(img)
    Y.append(imgClass)
    
X = np.array(X).astype(np.float32) / 255.0
Y = np.array(Y)

valLen = int(len(X)*0.2)
X_train = X[valLen:]
Y_train = Y[valLen:]

X_val = X[:valLen]
Y_val = Y[:valLen]

print("X_train,Y_train",X_train.shape,Y_train.shape)
print("X_val.shape,Y_val.shape",X_val.shape,Y_val.shape)

# View train images:
fig = plt.figure(figsize=(20,7))
for i,img in enumerate(X_train[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(img)
# Create model.

from keras.models import Sequential
from keras.layers import Input,Conv2D,MaxPool2D,Dropout,Activation,BatchNormalization,GlobalAveragePooling2D,Dense

model = Sequential()

model.add(Conv2D(32,(3,3),padding="same",activation="relu",use_bias=False,input_shape=outputImgShape))
model.add(MaxPool2D(padding="same"))

model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding="same",activation="relu",use_bias=False))
model.add(MaxPool2D(padding="same"))

model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding="same",activation="relu",use_bias=False))
model.add(MaxPool2D(padding="same"))
model.add(Dropout(rate=0.3))

model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding="same",activation="relu",use_bias=False))
model.add(MaxPool2D(padding="same"))
model.add(Dropout(rate=0.3))

model.add(BatchNormalization())
model.add(Conv2D(512,(3,3),padding="same",activation="relu",use_bias=False))
model.add(MaxPool2D(padding="same"))
model.add(Dropout(rate=0.3))

model.add(GlobalAveragePooling2D())
model.add(Dense(1,activation="sigmoid"))

model.summary()

model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

epochs = 50
batch_size = 16
from keras.callbacks import ReduceLROnPlateau

reduceLRCallback = ReduceLROnPlateau(factor=0.1,patience=5)
hist = model.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,callbacks=[reduceLRCallback],validation_data=(X_val,Y_val))
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))

ax[0].plot(hist.history["loss"],label="loss")
ax[0].plot(hist.history["val_loss"],label="val_loss")
ax[0].legend(loc="upper left")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss value")

ax[1].plot(hist.history["acc"],label="acc")
ax[1].plot(hist.history["val_acc"],label="val_acc")
ax[1].legend(loc="upper left")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Acc value")

plt.show()

testImgNames = sorted([int(imgName.split(".")[0]) for imgName in os.listdir(testHome)])

X_test = []

for imgName in tqdm(testImgNames):
    img = readImage(testHome+"/"+str(imgName)+".jpg",outputImgShape)
    X_test.append(img)
    
X_test = np.array(X_test).astype(np.float32) / 255.0
print("X_test",X_test.shape)
preds = model.predict(X_test)
#preds[preds>0.5] = 1
#preds[preds<=0.5] = 0
for i,imgName in tqdm(enumerate(testImgNames)):
    imgName = imgName
    prob = float(preds[i,0])
    print(str(imgName)+","+str(prob))


