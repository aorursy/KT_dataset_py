import os

import pandas as pd



# function to get all flowers into a dataframe

def GetFlowerDataFrame():

    dfColumns = ['FileLoc','Type'];

    flowerDataFrame = pd.DataFrame([], columns=dfColumns)

    for sub in os.listdir('../input/flowers/flowers'):

        if sub == ".DS_Store":

            continue

        for subsub in os.listdir('../input/flowers/flowers/'+sub):

            if subsub == ".DS_Store":

                continue

            if subsub.endswith(".jpg"):

                flowerDataFrame = flowerDataFrame.append(pd.DataFrame([['../input/flowers/flowers/'+sub+'/'+subsub, sub]], columns = dfColumns))

    flowerDataFrame.to_csv('FlowerDataList.csv', sep=',', index=False);

    return flowerDataFrame

print("Done cataloging flowers")

import keras

from keras import models

from keras import layers

from keras import optimizers

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import cv2

import matplotlib.pyplot as plt

import random

import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import OneHotEncoder



# loads images and makes a list of labels

def LoadImagesAndLabels(fileLocations, labels, imageSizeParameters):

    images = []

    for fileLocation in fileLocations:

        #print("Opening image: " + fileLocation)

        #os.call('clear' if os.name =='posix' else 'cls') 

        # normalize images to be the same size

        images.append(

            cv2.resize(cv2.imread(fileLocation), (imageSizeParameters["xSize"], imageSizeParameters["ySize"]), interpolation=cv2.INTER_CUBIC)

        )

    return [images, labels]





## read in data

flowerMetaData = GetFlowerDataFrame();

#flowerMetaData = pd.read_csv;

#flowerMetaData.set_index('FileLoc', inplace=True)

imageParameters = {

    "xSize": 150,

    "ySize": 150

}

[flowerImages, flowerLabels] =  LoadImagesAndLabels(flowerMetaData['FileLoc'], list(flowerMetaData['Type']), imageParameters)

print("Done loading images")
## segment data into training and test set

batchSize = 32

percTraining = 0.8

nrow = len(flowerLabels)

idx = np.arange(0, nrow-1, 1)

random.shuffle(idx)

trainIdx = idx[0:int(np.ceil(nrow*percTraining))]

testIdx = idx[int(np.ceil(nrow*percTraining)+1):nrow-1]

trImgs = np.array([flowerImages[i] for i in trainIdx])

trLblsStr = [flowerLabels[i] for i in trainIdx]

teImgs = np.array([flowerImages[i] for i in testIdx])

teLblsStr = [flowerLabels[i] for i in testIdx]

# one hot encode output labels

trLbls = np.array(pd.get_dummies(trLblsStr))

teLbls = np.array(pd.get_dummies(teLblsStr))

# check proportions for the training set

print(np.bincount(np.argmax(trLbls, axis=1)))



# preprocess images further using ImageDataGenerator

# scale images with perturbations

trPP = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# just scale test set of images

tePP = ImageDataGenerator(rescale=1./255)

trDatPP = trPP.flow(trImgs, trLbls, batch_size=batchSize)

teDatPP = tePP.flow(teImgs, teLbls, batch_size=batchSize)

print("Done preprocessing")
# function to calc metrics and do plots

def CalcMetricsAndMakePlots(model, modelName, trainingGenerator, numTraining, testData, testLabels, classLabels, numEpochs, batchSize):

    # fit the model

# fitting directly causing overfitting; going to use image generator to see if it fixes

#     modelLog = model.fit(

#         x = trainingData / 255.0, # scales image pixels [0:1]

#         y = trainingLabels,

#         epochs = numEpochs,

#         batch_size = batchSize,

#         verbose = 1,

#         validation_data = (testData / 255.0, testLabels)

#     )

    modelLog = model.fit_generator(

        generator = trainingGenerator,

        steps_per_epoch = numTraining // batchSize,

        epochs = numEpochs,

        validation_data = (testData / 255.0, testLabels),

        validation_steps = len(testLabels) // batchSize,

        verbose = 0

    )

    # do accuracy against number of epochs plot

    acc = modelLog.history['acc']

    validationAcc = modelLog.history['val_acc']

    epochs = range(1, numEpochs + 1)

    plt.plot(epochs, acc, label="Training accuracy")

    plt.plot(epochs, validationAcc, label='Validation accuracy')

    plt.title('Accuracy v. epochs for ' + modelName + ' model.' )

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()

    

    # do confusion matrix calculation calculations for test data

    modelPredDummys = model.predict(testData/255.0, verbose=1)

    modelPredLabels = classLabels[np.argmax(modelPredDummys, axis=1)]

    testLabelStrings = classLabels[np.argmax(testLabels, axis=1)]

    confMat = confusion_matrix(testLabelStrings, modelPredLabels)

    print(np.sum(confMat, axis=1))

    confMatProps = confMat / np.sum(confMat, axis=1)

    print(confMatProps)

    

    # do barchart of accuracy proportions

    classAcc = np.diag(confMatProps)

    print(classAcc)

    overallAcc = np.mean(classAcc)

    print(overallAcc)

    accVector = np.concatenate([classAcc, [overallAcc]])

    lblVector = np.concatenate([classLabels, ['overall']])

    print(accVector)

    print(lblVector)

    resultsMetricDataFrame = pd.DataFrame(

        [accVector], columns = lblVector

    )

    print(resultsMetricDataFrame)

    resultsMetricDataFrame.plot(kind="barh")

    plt.xlabel('Accuracy')

    plt.ylabel('Classes')

    # puts legend outside of plot axes

    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.title('Accuracy per class and overall for ' + modelName + ' model.' )

    



print("Done defining function")

    
## build model for LeNet-5

# define layers according to LeNet-5

numEpoch = 128

numCategories = len(np.unique(flowerLabels))

cnnLayers = []

cnnLayers.append(layers.Conv2D(6, (5, 5), activation='tanh' , 

                input_shape=(imageParameters['xSize'], imageParameters['ySize'], 3)))

cnnLayers.append(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))

cnnLayers.append(layers.Conv2D(16, (5, 5), activation='tanh' ))

cnnLayers.append(layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))

cnnLayers.append(layers.Flatten())

cnnLayers.append(layers.Dense(84, activation='tanh'))

cnnLayers.append(layers.Dense(numCategories, activation='softmax'))



mdlLeNet = models.Sequential()

for cnnLayer in cnnLayers:

    #print(cnnLayer)

    mdlLeNet.add(cnnLayer);



mdlLeNet.compile(loss=keras.losses.categorical_crossentropy,

            #optimizer=optimizers.Adam(lr=0.001),

            optimizer=optimizers.Adagrad(),

            metrics=['accuracy'])



CalcMetricsAndMakePlots(mdlLeNet, 'LeNet-5 architecture', trDatPP, len(trLbls), teImgs, teLbls, np.unique(flowerLabels), numEpoch, batchSize)

del mdlLeNet # deletes mdl from memory - weird convergence over time

del cnnLayers





## build model for AlexNet (2012)

np.random.seed(1)

# define layers according to VGGNet arch - ConvNet Configuration A in original paper

cnnLayers = []

cnnLayers.append(layers.Conv2D(filters=48, kernel_size=(11, 11), activation='relu', 

                               input_shape=(imageParameters['xSize'], imageParameters['ySize'], 3)))

cnnLayers.append(layers.MaxPooling2D((5, 5)))

cnnLayers.append(layers.Conv2D(128, (5, 5), activation='relu'))

cnnLayers.append(layers.MaxPooling2D((3, 3)))

cnnLayers.append(layers.Conv2D(192, (3, 3), activation='relu'))

cnnLayers.append(layers.Conv2D(192, (3, 3), activation='relu'))

cnnLayers.append(layers.Conv2D(128, (3, 3), activation='relu'))

cnnLayers.append(layers.Flatten())

cnnLayers.append(layers.Dropout(0.5))

cnnLayers.append(layers.Dense(2048, activation='relu'))

cnnLayers.append(layers.Dropout(0.5))

cnnLayers.append(layers.Dense(2048, activation='relu'))

cnnLayers.append(layers.Dense(numCategories, activation='softmax'))



mdlAlexNet = models.Sequential()

for cnnLayer in cnnLayers:

    print(cnnLayer)

    mdlAlexNet.add(cnnLayer);



mdlAlexNet.compile(loss=keras.losses.categorical_crossentropy,

            #optimizer=optimizers.Adam(lr=0.001),

            #optimizer=optimizers.Adagrad(),

            optimizer=optimizers.SGD(momentum=0.9, decay=0.0005),

            metrics=['accuracy'])



CalcMetricsAndMakePlots(mdlAlexNet, 'AlexNet architecture', trDatPP, len(trLbls), teImgs, teLbls, np.unique(flowerLabels), numEpoch, batchSize)

del mdlAlexNet # deletes mdl from memory - weird convergence over time

del cnnLayers

## build model with keras for VGGNet

np.random.seed(1)

# define layers according to VGGNet arch - ConvNet Configuration A in original paper

cnnLayers = []

cnnLayers.append(layers.Conv2D(32, (3, 3), activation='relu', 

                               input_shape=(imageParameters['xSize'], imageParameters['ySize'], 3)))

cnnLayers.append(layers.MaxPooling2D((2, 2)))

cnnLayers.append(layers.Conv2D(64, (3, 3), activation='relu'))

cnnLayers.append(layers.MaxPooling2D((2, 2)))

cnnLayers.append(layers.Conv2D(128, (3, 3), activation='relu'))

cnnLayers.append(layers.Conv2D(128, (3, 3), activation='relu'))

cnnLayers.append(layers.MaxPooling2D((2, 2)))

cnnLayers.append(layers.Conv2D(256, (3, 3), activation='relu'))

cnnLayers.append(layers.Conv2D(256, (3, 3), activation='relu'))

cnnLayers.append(layers.Conv2D(256, (3, 3), activation='relu'))

cnnLayers.append(layers.Conv2D(256, (3, 3), activation='relu'))

cnnLayers.append(layers.MaxPooling2D((2, 2)))

cnnLayers.append(layers.Flatten())

cnnLayers.append(layers.Dropout(0.5))

cnnLayers.append(layers.Dense(2048, activation='relu'))

cnnLayers.append(layers.Dropout(0.5))

cnnLayers.append(layers.Dense(2048, activation='relu'))

cnnLayers.append(layers.Dense(500, activation='relu'))

cnnLayers.append(layers.Dense(numCategories, activation='softmax'))



mdlVgg = models.Sequential()

for cnnLayer in cnnLayers:

    #print(cnnLayer)

    mdlVgg.add(cnnLayer);



mdlVgg.compile(loss=keras.losses.categorical_crossentropy,

            optimizer=optimizers.Adam(lr=0.001),

            #optimizer=optimizers.Adagrad(),

            metrics=['accuracy'])



CalcMetricsAndMakePlots(mdlVgg, 'VGG CNN - Architecture A', trDatPP, len(trLbls), teImgs, teLbls, np.unique(flowerLabels), numEpoch, batchSize)

del mdlVgg # deletes mdl from memory - weird convergence over time

del cnnLayers