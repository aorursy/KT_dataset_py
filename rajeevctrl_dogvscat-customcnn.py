# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
home = "../input"
trainHome = home+"/train"
testHome = home+"/test"
print("Total train images:",len(os.listdir(trainHome)))
print("Total test images:",len(os.listdir(testHome)))
classes = [1 if fileName.split(".")[0]=="dog" else 0 for fileName in os.listdir(trainHome)]

print("Total train examples:",len(classes))
print("Total dogs:",np.sum(classes))
print("Total cats:",len(classes)-np.sum(classes))
def fetchImage(imagePath,imgShape):
    frame = cv2.imread(imagePath,cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame,(imgShape[0],imgShape[1]))
    #frame = frame if isNormalized==False else frame/255.0
    return frame

def fetchImgData(home,imageShape,isTrain=True,isNormalized=False):
    imgNames = os.listdir(home)
    X = []
    Y = []
    for imgName in tqdm(imgNames):
        frame = fetchImage(home+"/"+imgName,imgShape)
        X.append(frame)
        if isTrain==True:
            imgClassName = imgName.split(".")[0]
            imgClass = 1 if imgClassName=="dog" else 0
            Y.append(imgClass)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def fetchBatch(home,imgNamesList,batch_num,batch_size,imgShape):
    imgNames = imgNamesList[batch_num*batch_size : (batch_num+1)*batch_size]
    X = []
    Y = []
    for imgName in imgNames:
        frame = fetchImage(home+"/"+imgName,imgShape)
        X.append(frame)
        imgClassName = imgName.split(".")[0]
        imgClass = 1 if imgClassName=="dog" else 0
        Y.append(imgClass)
    return np.array(X), np.array(Y)

def shuffle(lst):
    result = []
    first = lst[0]
    indicies = np.random.permutation(np.arange(len(lst[0])))
    for l in lst:
        l = np.array(l)[indicies]
        result.append(l)
    return result
imgShape = (200,200,3)
#X_train,Y_train = fetchImgData(trainHome,imgShape,isTrain=True,isNormalized=True)
trainImgNames = shuffle([os.listdir(trainHome)])[0]
X_train,Y_train = fetchBatch(trainHome,trainImgNames,0,5,imgShape)
fix,ax = plt.subplots(nrows=1,ncols=5,figsize=(20,3))
for i,img in enumerate(X_train):
    ax[i].imshow(img)
    
    
from keras.models import Sequential,Model
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense,Input,Flatten
from keras.utils import to_categorical
modelResnet = ResNet50(include_top=False,weights="imagenet",input_shape=imgShape)
resnetLayers = modelResnet.layers
print("Num layers Resnet50:",len(resnetLayers))

#print([layer.trainable for layer in resnetLayers])
#for layer in resnetLayers[:-10]:
#    layer.trainable=False
#print([layer.trainable for layer in resnetLayers])
# RESNET model.
input_ = Input(shape=imgShape)
out = modelResnet(input_)
out = Flatten()(out)
dense = Dense(2)(out)

model = Model(inputs=[input_],outputs=[dense])
print("Final model summary")
print(model.summary())

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
# CUSTOM Model.

from keras.layers import Conv2D,MaxPool2D,Dense,Activation,GlobalAvgPool2D,BatchNormalization
from keras.layers import Input,Dropout


input_ = Input(shape=imgShape)
layer = Conv2D(32,(3,3),padding="same",activation="relu")(input_)
layer = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same")(layer)
layer = Dropout(0.3)(layer)

layer = BatchNormalization()(layer)
layer = Conv2D(64,(3,3),padding="same",activation="relu")(layer)
layer = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same")(layer)
layer = Dropout(0.3)(layer)

layer = BatchNormalization()(layer)
layer = Conv2D(128,(3,3),padding="same",activation="relu")(layer)
layer = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same")(layer)
layer = Dropout(0.3)(layer)

layer = BatchNormalization()(layer)
layer = Conv2D(256,(3,3),padding="same",activation="relu")(layer)
layer = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same")(layer)
layer = Dropout(0.3)(layer)

layer = BatchNormalization()(layer)
layer = Conv2D(512,(3,3),padding="same",activation="relu")(layer)
layer = MaxPool2D(pool_size=(2,2),strides=(2,2),padding="same")(layer)
layer = Dropout(0.3)(layer)

layer = GlobalAvgPool2D()(layer)
out = Dense(2,activation="softmax")(layer)

model = Model(inputs = input_, outputs=out)
print("Final model summary")
print(model.summary())

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
imgNames = shuffle([os.listdir(trainHome)])[0]
valLen = int(len(imgNames)*0.1)

valImgNames = imgNames[:valLen]
trainImgNames = imgNames[valLen:]
print("Total train images:",len(trainImgNames))
print("Total val images:",len(valImgNames))

batch_size=16
epochs = 10
num_batches = int(len(trainImgNames)/batch_size)

X_val,Y_val = fetchBatch(trainHome,valImgNames,0,len(valImgNames),imgShape)
X_val = X_val.astype(np.float32)/ 255.0
Y_val = to_categorical(Y_val,num_classes=2)
hist={
    "loss":[],
    "acc":[],
    "val_loss":[],
    "val_acc":[]
}

for i_epoch in range(epochs):
    for i_batch in range(num_batches):
        X,Y = fetchBatch(trainHome,trainImgNames,i_batch,batch_size,imgShape)
        X = X.astype(np.float32) / 255.0
        Y = to_categorical(Y,num_classes=2)
        model.train_on_batch(X,Y)
        loss,acc = model.evaluate(X,Y,verbose=0)
        sys.stdout.write("\r{}/{} Epoch {}/{} loss: {} acc: {}".format(i_batch+1,num_batches,i_epoch+1,epochs,loss,acc))
    val_loss,val_acc = model.evaluate(X_val,Y_val,verbose=0)
    sys.stdout.write("Epoch {}/{} loss: {} acc: {} val_loss: {} val_acc: {}".format(i_epoch+1,epochs,loss,acc,val_loss,val_acc))
    sys.stdout.write("\n")
    hist["loss"].append(loss)
    hist["acc"].append(acc)
    hist["val_loss"].append(val_loss)
    hist["val_acc"].append(val_acc)
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(15,5))

ax[0].plot(hist["loss"],label="loss")
ax[0].plot(hist["val_loss"],label="val_loss")
ax[0].legend(loc="upper left")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss values")

ax[1].plot(hist["acc"],label="acc")
ax[1].plot(hist["val_acc"],label="val_acc")
ax[1].legend(loc="upper left")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Acc values")


testImgNames = sorted(os.listdir(testHome))
print("Total test imgs",len(testImgNames))
print(testImgNames[:10])

X_test = [fetchImage(testHome+"/"+testImgName,imgShape) for testImgName in tqdm(testImgNames)]
X_test = np.array(X_test,dtype=np.float32) / 255.0
X_test.shape
preds = model.predict(X_test)
print(preds[0,0],preds[0,1])
fig = plt.figure(figsize=(25,20))
for i,img in enumerate(X_test[:50]):
    plt.subplot(5,10,i+1)
    plt.imshow(img)
    #print("Cat:{0:.2f} Dog:{0:.2f}".format(preds[i,0],preds[i,1]),"Dog:{0:.2f}".format(preds[i,1]))
    sortProbs = np.argsort(preds[i])
    class_ , prob = ("Dog",preds[i,1]) if sortProbs[1]==1 else ("Cat",preds[i,0])
    text = class_+":{0:.2f}".format(prob)
    #text = "Cat:{0:.2f}".format(preds[i,0])+" Dog:{0:.2f}".format(preds[i,1])
    #print(text)
    plt.title(text)
    #plt.text(10,10,text,fontsize=12)
    #plt.show()


