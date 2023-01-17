# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2
import math
from operator import itemgetter
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, BatchNormalization, GlobalMaxPooling2D, Lambda, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
inputDir = os.path.join(os.getcwd(),"..","input")

# Any results you write to the current directory are saved as output.
trainDirLen = len(os.listdir(os.path.join(inputDir,"train")))
actTestDirLen = len(os.listdir(os.path.join(inputDir,"test")))
trainDirImgList = os.listdir(os.path.join(inputDir,"train"))
trainDir = os.path.join(inputDir,"train")
testDir= os.path.join(inputDir,"test")

trainGenerator = None
validGenerator = None
testGenerator = None
numClasses = 2
labelDict = {
'cat':0,
'dog':1
}
def convertToOHE(x):
    try:
        return labelDict[x]
    except:
        return None

def extractLabels(dataDir):
    fileList = os.listdir(dataDir)
    nameDf = pd.DataFrame(fileList, columns=["filename"])
    nameDf["basename"] = nameDf["filename"].apply(lambda x: os.path.splitext(x)[0])
    nameDf = nameDf.merge(nameDf.basename.apply(lambda x:pd.Series({'left':x.split('.')[0],'right':x.split('.')[1]})),left_index=True,right_index=True)
    nameDf["labels"] = nameDf.left.apply(convertToOHE)
    labels = to_categorical(nameDf["labels"],num_classes=2)
    return labels
def createGenerator(fileList,labels,batch_size,imgDir):
    global numClasses
    fileListLength = len(fileList)
    labelContainer = np.zeros((batch_size,numClasses))
    maxX=0
    maxY=0
    while True:
        randArr = np.random.permutation(fileListLength)
        imageList = list()
        maxX=maxY=0
        for i in range(batch_size):
            fi = os.path.join(imgDir,fileList[randArr[i]])
            image = cv2.imread(fi,cv2.IMREAD_GRAYSCALE)
            labelContainer[i] = labels[randArr[i]]
            imageList.append(image)
            if image.shape[0]>maxX:
                maxX= image.shape[0]
            if image.shape[1]>maxY:
                maxY = image.shape[1]
        imageContainer = np.zeros((batch_size,maxX,maxY))
        for i in range(batch_size):
            image = imageList[i]
            imageContainer[i,:image.shape[0],:image.shape[1]] = image
        imageContainer = np.expand_dims(imageContainer,axis=3)
        
        yield np.array(imageContainer),labelContainer


def prepareGenerator(dataDir,labels, batch_size):
    global trainGenerator
    global validGenerator
    global testGenerator
    fileList = os.listdir(dataDir)
    fileListLen = len(fileList)
    index = np.random.permutation(fileListLen)
    trainMarker = math.floor(index.size*(0.6))
    validateMarker = math.floor(index.size*(0.8))
    trainIndex = index[0:trainMarker]
    validateIndex = index[trainMarker:validateMarker]
    testIndex = index[validateMarker:]
    trainFiles = itemgetter(*trainIndex)(fileList)
    trainLabels = labels[trainIndex,...]
    validateFiles = itemgetter(*validateIndex)(fileList)
    validateLabels = labels[validateIndex,...]
    testFiles = itemgetter(*testIndex)(fileList)
    testLabels = labels[testIndex,...]
    trainGenerator = createGenerator(trainFiles,trainLabels,batch_size,dataDir)
    validGenerator = createGenerator(validateFiles,validateLabels,batch_size,dataDir)
    testGenerator = createGenerator(testFiles, testLabels, batch_size,dataDir)
def get_callbacks(filepath, patience=5):
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    es = EarlyStopping(monitor='val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.0001)
    return [es,checkpoint,reduce_lr]
import matplotlib.pyplot as plt
from keras.models import load_model
labels = extractLabels(trainDir)
prepareGenerator(trainDir,labels,16)
def getModel():
    inp = Input(shape=(None,None,1))
    x = Lambda(lambda y:y/255.0)(inp)
    x = Conv2D(8,(7,7), activation="elu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(16,(5,5), activation="elu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(3,3), activation="elu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64,(1,1))(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(16, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(8, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(4, activation="relu")(x)
    x = BatchNormalization()(x)
    out = Dense(2, activation="softmax")(x)
    model = Model(inputs = inp, outputs=out)
    model.compile(optimizer = "sgd", loss = "categorical_crossentropy",metrics=["accuracy"])
    model.summary()
    return model

modelSavePath = os.path.join(os.getcwd(),".model_weights_commit.hdf5")
callbacks = get_callbacks(modelSavePath,8)
gmodel = getModel()
history = gmodel.fit_generator(trainGenerator,steps_per_epoch=8,epochs=40, verbose=1, callbacks=callbacks, validation_data=validGenerator, validation_steps=8, )
gmodel = load_model(modelSavePath)
scores = gmodel.evaluate_generator(testGenerator, steps=32)
print(scores)
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_acc'])
