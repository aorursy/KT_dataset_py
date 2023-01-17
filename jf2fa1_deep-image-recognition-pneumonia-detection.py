!pip install keras
!pip install pydot
!pip install graphviz
# IPython display functions
import IPython
from IPython.display import display, HTML, SVG, Image

# General Plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')
plt.rcParams['figure.figsize'] = [10, 6] ## plot size
plt.rcParams['axes.linewidth'] = 2.0 #set the value globally

## notebook style and settings
display(HTML("<style>.container { width:90% !important; }</style>"))
display(HTML("<style>.output_png { display: table-cell; text-align: center; vertical-align: middle; } </style>"))
display(HTML("<style>.MathJax {font-size: 100%;}</style>"))

# For changing background color
def set_background(color):
    script = ( "var cell = this.closest('.code_cell');" "var editor = cell.querySelector('.input_area');" "editor.style.background='{}';" "this.parentNode.removeChild(this)" ).format(color)
    display(HTML('<img src onerror="{}">'.format(script)))
import os
import sys
import random
import numpy as np
import pandas as pd
from os import walk

# Metrics
from sklearn.metrics import *

# Keras library for deep learning
# https://keras.io/
import tensorflow as tf
import keras
from keras.datasets import mnist # MNIST Data set
from keras.models import Sequential # Model building
from keras.layers import * # Model layers
from keras.preprocessing.image import *
from sklearn.model_selection import train_test_split

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import glob
import h5py
import shutil
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras import backend as K
color = sns.color_palette()
%matplotlib inline
def displayConfusionMatrix(confusionMatrix, precisionNegative, precisionPositive, recallNegative, recallPositive, title):
    # Set font size for the plots. You can ignore this line.
    PLOT_FONT_SIZE = 14
    
    # Set plot size. Please ignore this line
    plt.rcParams['figure.figsize'] = [5, 5]
    
    # Transpose of confusion matrix to align the plot with the actual precision recall values. Please ignore this as well.
    confusionMatrix = np.transpose(confusionMatrix)
    
    # Plotting the confusion matrix
    plt.imshow(confusionMatrix, interpolation='nearest',cmap=plt.cm.Blues, vmin=0, vmax=100)
    
    
    # Setting plot properties. You should ignore everything from here on.
    xticks = np.array([-0.5, 0, 1,1.5])
    plt.gca().set_xticks(xticks)
    plt.gca().set_yticks(xticks)
    plt.gca().set_xticklabels(["", "Healthy\nRecall=" + str(recallNegative) , "Pneumonia\nRecall=" + str(recallPositive), ""], fontsize=PLOT_FONT_SIZE)
    plt.gca().set_yticklabels(["", "Healthy\nPrecision=" + str(precisionNegative) , "Pneumonia\nPrecision=" + str(precisionPositive), ""], fontsize=PLOT_FONT_SIZE)
    plt.ylabel("Predicted Class", fontsize=PLOT_FONT_SIZE)
    plt.xlabel("Actual Class", fontsize=PLOT_FONT_SIZE)
    plt.title(title, fontsize=PLOT_FONT_SIZE)
        
    # Add text in heatmap boxes
    for i in range(2):
        for j in range(2):
            text = plt.text(j, i, confusionMatrix[i][j], ha="center", va="center", color="white", size=15) ### size here is the size of text inside a single box in the heatmap
            
    plt.show()
def calculateMetricsAndPrint(predictions, predictionsProbabilities, actualLabels):
    # Convert label format from [0,1](label 1) and [1,0](label 0) into single integers: 1 and 0.
    actualLabels = [item[1] for item in actualLabels]
    
    # Get probabilities for the class with label 1. That is all we need to compute AUCs. We don't need probabilities for class 0.
    predictionsProbabilities = [item[1] for item in predictionsProbabilities]
    
    # Calculate metrics using scikit-learn functions. The round function is used to round the numbers up to 2 decimal points.
    accuracy = round(accuracy_score(actualLabels, predictions) * 100, 2)
    precisionNegative = round(precision_score(actualLabels, predictions, average = None)[0] * 100, 2)
    precisionPositive = round(precision_score(actualLabels, predictions, average = None)[1] * 100, 2)
    recallNegative = round(recall_score(actualLabels, predictions, average = None)[0] * 100, 2)
    recallPositive = round(recall_score(actualLabels, predictions, average = None)[1] * 100, 2)
    auc = round(roc_auc_score(actualLabels, predictionsProbabilities) * 100, 2)
    confusionMatrix = confusion_matrix(actualLabels, predictions)
    
    # Print metrics. .%2f prints a number upto 2 decimal points only.
    print("------------------------------------------------------------------------")
    print("Accuracy: %.2f\nPrecisionNegative: %.2f\nPrecisionPositive: %.2f\nRecallNegative: %.2f\nRecallPositive: %.2f\nAUC Score: %.2f" % 
          (accuracy, precisionNegative, precisionPositive, recallNegative, recallPositive, auc))
    print("------------------------------------------------------------------------")
    
    print("+ Printing confusion matrix...\n")
    # Display confusion matrix
    displayConfusionMatrix(confusionMatrix, precisionNegative, precisionPositive, recallNegative, recallPositive, "Confusion Matrix")
    
    print("+ Printing ROC curve...\n")
    # ROC Curve
    plt.rcParams['figure.figsize'] = [16, 8]
    FONT_SIZE = 16
    falsePositiveRateDt, truePositiveRateDt, _ = roc_curve(actualLabels, predictionsProbabilities)
    plt.plot(falsePositiveRateDt, truePositiveRateDt, linewidth = 5, color='black')
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xlabel("False Positive Rate", fontsize=FONT_SIZE)
    plt.ylabel("True Positive Rate", fontsize=FONT_SIZE)
    plt.show()
    
    return auc
def getKagglePredictions(model, kaggleData, filename):
    print("+ Writing kaggle test results in : ../input/output/%s..." % filename)
    predictions = model.predict(kaggleData)
    predictionProbs = [item[1] for item in predictions]
        
    # Store predictions for kaggle
    outputFile = open("../input/output/" + str(filename), "w")
    outputFile.write("Id,Prediction\n")
    for i in range(0, len(predictionProbs)):
        outputFile.write(str(i + 1) + "," + str(predictionProbs[i]) + "\n")
    
    outputFile.close()
def calculateClasswiseTopNAccuracy(actualLabels, predictionsProbs, TOP_N):
    """
    TOP_N is the top n% predictions you want to use for each class
    """

    discreteActualLabels = [1 if item[1] > item[0] else 0 for item in actualLabels]
    discretePredictions = [1 if item[1] > item[0] else 0 for item in predictionsProbs]
    predictionProbsTopNHealthy, predictionProbsTopNPneumonia = [item[0] for item in predictionsProbs], [item[1] for item in predictionsProbs]
    predictionProbsTopNHealthy = list(reversed(sorted(predictionProbsTopNHealthy)))[:int(len(predictionProbsTopNHealthy) * TOP_N / 100)][-1]
    predictionProbsTopNPneumonia = list(reversed(sorted(predictionProbsTopNPneumonia)))[:int(len(predictionProbsTopNPneumonia) * TOP_N / 100)][-1]

    # Calculate accuracy for both classes
    accuracyHealthy = []
    accuracyPneumonia = []
    for i in range(0, len(discretePredictions)):
        if discretePredictions[i] == 1:
            # Pneumonia
            if predictionsProbs[i][1] > predictionProbsTopNPneumonia:
                accuracyPneumonia.append(int(discreteActualLabels[i]) == 1)
        else:
            # Healthy
            if predictionsProbs[i][0] > predictionProbsTopNHealthy:
                accuracyHealthy.append(int(discreteActualLabels[i]) == 0)

    accuracyHealthy = round((accuracyHealthy.count(True) * 100) / len(accuracyHealthy), 2)
    accuracyPneumonia = round((accuracyPneumonia.count(True) * 100) / len(accuracyPneumonia), 2)
    return accuracyHealthy, accuracyPneumonia
# Load normal images
normalImagesPath = "../input/deep-image-normal/normal"
normalImageFiles = []
for(_,_,files) in walk(normalImagesPath):
    normalImageFiles.extend(files)


# Load pneumonia images
pneumoniaImagesPath = "../input/deep-image-normal/pneumonia/pneumonia"
pneumoniaImageFiles = []
for(_,_,files) in walk(pneumoniaImagesPath):
    pneumoniaImageFiles.extend(files)
    
random.shuffle(pneumoniaImageFiles)
pneumoniaImageFiles = pneumoniaImageFiles[:len(normalImageFiles)]
print("Normal X-ray images: %d\nPneumonia X-ray images: %d" % (len(normalImageFiles), len(pneumoniaImageFiles)))
imagesData = []
imagesLabels = []

for file in normalImageFiles:
    fullPath = normalImagesPath + "/" + file
    if os.path.exists(fullPath) == False:
            continue
    imageData = load_img(normalImagesPath + "/" + file, color_mode = "grayscale") # load_img function comes from keras library when we do "from keras.preprocessing.image import *"
    imageArray = img_to_array(imageData) / 255.0
    
    imagesData.append(imageArray)
    imagesLabels.append(0)
    

for file in pneumoniaImageFiles:
    fullPath = pneumoniaImagesPath + "/" + file
    if os.path.exists(fullPath) == False:
            continue
            
    imageData = load_img(pneumoniaImagesPath + "/" + file, color_mode = "grayscale") # load_img function comes from keras library when we do "from keras.preprocessing.image import *"
    imageArray = img_to_array(imageData) / 255.0
    
    imagesData.append(imageArray)
    imagesLabels.append(1)

imagesData = np.array(imagesData)
imagesLabels = keras.utils.to_categorical(imagesLabels)
print("Input data shape: %s" % (imagesData.shape,))
testImagesPath = "d../input/deep-image-normal/test/test"
testImageFiles = []
for(_,_,files) in walk(testImagesPath):
    testImageFiles.extend(files)
testImageFiles = list(sorted(testImageFiles))
    
kaggleTestImages = []
for file in testImageFiles:
    fullPath = testImagesPath + "/" + file
    if os.path.exists(fullPath) == False:
        continue
    imageData = load_img(testImagesPath + "/" + file, color_mode = "grayscale") # load_img function comes from keras library when we do "from keras.preprocessing.image import *"
    imageArray = img_to_array(imageData) / 255.0
    
    kaggleTestImages.append(imageArray)
    
kaggleTestImages = np.array(kaggleTestImages)
print("Number of test images: %d" % len(kaggleTestImages))
def trainTestSplit(data, labels):
    """
    80-20 train-test data split
    """
    trainData, trainLabels, testData, testLabels = [], [], [], []
    for i in range(0, len(data)):
        if i % 5 == 0:
            testData.append(data[i])
            testLabels.append(labels[i])
        else:
            trainData.append(data[i])
            trainLabels.append(labels[i])
            
    return np.array(trainData), np.array(testData), np.array(trainLabels), np.array(testLabels)
# Split data into 80% training and 20% testing
trainData, testData, trainLabels, testLabels = trainTestSplit(imagesData, imagesLabels)
def createParameterizedConvolutionalNeuralNetwork(trainImages, numLayers, numFilters, kernelSize, maxPooling, dropoutValue, learningRate, numClasses):
    # Create model object
    model = Sequential()
    
    model.add(Conv2D(numFilters, kernel_size=(kernelSize, kernelSize),
                       activation='relu', padding = 'same',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(maxPooling, maxPooling)))
    model.add(Dropout(dropoutValue))
    
    while numLayers > 1:
        model.add(Conv2D(numFilters, kernel_size=(kernelSize, kernelSize),
                     activation='relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size=(maxPooling, maxPooling)))
        model.add(Dropout(dropoutValue))
        
        numLayers = numLayers - 1
        
    # Flatten & Final dropout
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropoutValue))
    model.add(Dense(numClasses, activation='softmax'))

    # Compile model.
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learningRate),
                  metrics=['accuracy'])
    # Return model
    return model

def createNuancedConvolutionalNeuralNetwork(trainImages, numClasses):

        # Create model object
    model = Sequential()
    

    model.add(Conv2D(filters = 32, kernel_size=(3, 3),
                     activation='relu', padding = 'Valid',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(filters = 64, kernel_size=(5, 5),
                     activation='relu', padding = 'Valid',
                     input_shape=trainImages.shape[1:]))
    model.add(Conv2D(filters = 64, kernel_size=(5, 5),
                     activation='relu', padding = 'Valid',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.3))
     
    # Second layer with diffefiltersrent parameters
    model.add(Conv2D(filters = 128, kernel_size=(7, 7),
                     activation='relu', padding = 'Valid',
                     input_shape=trainImages.shape[1:]))
    model.add(Conv2D(filters = 128, kernel_size=(7, 7),
                     activation='relu', padding = 'Valid',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
    model.add(Dropout(0.5))
  
    # Flatten & Final dropout
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(numClasses, activation='softmax'))

    # Compile model. 
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    
    # Return model
    return model
def cnn0(trainImages, numClasses):

        # Create model object
    model = Sequential()
    

    model.add(Conv2D(filters = 16, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(Conv2D(filters = 16, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    

    model.add(SeparableConv2D(filters = 32, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 32, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ##model.add(Dropout(0.3))
    

    model.add(SeparableConv2D(filters = 64, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 64, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    ##model.add(Dropout(0.3))
    

    model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    

    model.add(SeparableConv2D(filters = 256, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 256, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
  
    # Flatten & Final dropout
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(numClasses, activation='softmax'))

    # Compile model. You can skip this line.
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-5),
                  metrics=['accuracy'])
                  
    # Return model
    return model


def cnn1(trainImages, numClasses):

        # Create model object
    model = Sequential()
    
    # Conv Layers
    model.add(Conv2D(filters = 64, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(Conv2D(filters = 64, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(SeparableConv2D(filters = 256, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 256, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 256, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(SeparableConv2D(filters = 512, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 512, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(SeparableConv2D(filters = 512, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
  
    # Flatten & Final dropout
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses, activation='softmax'))

    # Compile model. 
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-5),
                  metrics=['accuracy'])
                  
    # Return model
    return model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
def cnn2(trainImages, numClasses):

        # Create model object
    model = Sequential()
    
    # Add the first layer with dropout
    model.add(Conv2D(filters = 16, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(Conv2D(filters = 16, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(SeparableConv2D(filters = 32, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 32, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(SeparableConv2D(filters = 64, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 64, kernel_size=(5, 5),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(SeparableConv2D(filters = 128, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 128, kernel_size=(5, 5),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    
    model.add(SeparableConv2D(filters = 256, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 256, kernel_size=(5, 5),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 256, kernel_size=(7, 7),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(SeparableConv2D(filters = 512, kernel_size=(3, 3),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 512, kernel_size=(5, 5),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(SeparableConv2D(filters = 512, kernel_size=(7, 7),
                     activation='relu', padding = 'Same',
                     input_shape=trainImages.shape[1:]))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    
  
    # Convolutional layers are done
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(numClasses, activation='softmax'))

    # Compile model. You can skip this line.
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.0001, decay=1e-5),
                  metrics=['accuracy'])
                  
    # Return model
    return model
set_background('#fce53a')

dataAugmentation = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0,
    zoom_range=0.2)
set_background('#fce53a')


numLayers = 5 
numFilters = 32
kernelSize = 5 
dropoutValue = 0.4
maxPooling = 2 
numClasses = 2 
batchSize = 16 
learningRate = .00001 
epochs = 1 
USE_DATA_AUGMENTATION = False 




dataAugmentation.fit(trainData) 
# Create model
parameterizedModel = createParameterizedConvolutionalNeuralNetwork(trainData, numLayers, numFilters, kernelSize, maxPooling, dropoutValue, learningRate, numClasses = 2)
print("+parameterized model has been created...")
cnn0 = cnn0(imagesData, numClasses = 2)
cnn1 = cnn1(imagesData, numClasses = 2)
cnn2 = cnn2(imagesData, numClasses = 2)
print("+parameterized model has been created...")
model = cnn2
bestAcc = 0.0
bestEpoch = 0
bestAccPredictions, bestAccPredictionProbabilities = [], []

print("+ Starting training" )
print("-----------------------------------------------------------------------\n")
for epoch in range(epochs):
    
    #################################################### Model Training ###############################################################
    if USE_DATA_AUGMENTATION == True:
        # Use data augmentation in alternate epochs
        if epoch % 2 == 0:
            # Alternate between training with and without augmented data. 
            model.fit_generator(dataAugmentation.flow(trainData, trainLabels, batch_size=batchSize),
                        steps_per_epoch=len(trainData) / batchSize, epochs=1, verbose = 2)
        else:
            model.fit(trainData, trainLabels, batch_size=batchSize, epochs=1, verbose = 2)
    else:
        # Do not use data augmentation
        model.fit(trainData, trainLabels, batch_size=batchSize, epochs=1, verbose = 2)
    
    
    #################################################### Model Testing ###############################################################
    # Calculate test accuracy
    accuracy = round(model.evaluate(testData, testLabels)[1] * 100, 3)
    predictions = model.predict(testData)
    print("+ Test accuracy at epoch %d is: %.2f" % (epoch, accuracy))
    
    if accuracy > bestAcc:
        bestEpoch = epoch
        bestAcc = accuracy
        bestAccPredictions = [1 if item[1] > item[0] else 0 for item in predictions]
        bestAccPredictionProbabilities = predictions
        
        ##################################### Store predictions for kaggle ###########################################################
        #kaggleResultsFileName = "epoch-" + str(epoch) + "-resultscnn0.csv"
        #getKagglePredictions(model, kaggleTestImages, kaggleResultsFileName)
        ##############################################################################################################################
    print('\n')
print("------------------------------------------------------------------------")


##################################################### Printing best metrics ##########################################################
# Get more metrics for the best performing epoch
print("\n*** Printing our best validation results that we obtained in epoch %d ..." % bestEpoch)
calculateMetricsAndPrint(bestAccPredictions, bestAccPredictionProbabilities, testLabels)
#model.save('my_model.h5')
# Different Output view for PPT
!pip install keras_sequential_ascii
from keras_sequential_ascii import keras2ascii
keras2ascii(cnn2)
%load_ext tensorboard
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

################################## Does not work with Sample Dataset ###########################
topNValues = [5, 10, 20, 30]
##############################################################################################################

#accuraciesHealthy, accuraciesPneumonia = [], []
#for topn in topNValues:
#    accuracyHealthy, accuracyPneumonia = calculateClasswiseTopNAccuracy(testLabels, bestAccPredictionProbabilities, topn)
#    accuraciesHealthy.append(accuracyHealthy)
#    accuraciesPneumonia.append(accuracyPneumonia)
    
#    print("+ Accuracy for top %d percent predictions for healthy: %.2f, pneumonia: %.2f" % (topn, accuracyHealthy, accuracyPneumonia))
    
# Plot results
#x = np.arange(len(accuraciesHealthy))
#plt.plot(x, accuraciesHealthy, linewidth = 3, color = '#e01111')
#scatterHealthy = plt.scatter(x, accuraciesHealthy, marker = 's', s = 100, color = '#e01111')
#plt.plot(x, accuraciesPneumonia, linewidth = 3, color = '#0072ff')
#scatterPneumonia = plt.scatter(x, accuraciesPneumonia, marker = 'o', s = 100, color = '#0072ff')
#plt.xticks(x, topNValues, fontsize = 15)
#plt.yticks(fontsize = 15)
#plt.xlabel("Top N%", fontsize = 15)
#plt.ylabel("Accuracy", fontsize = 15)
#plt.legend([scatterHealthy, scatterPneumonia], ["Accuracy for Healthy", "Accuracy for Pneumonia"], fontsize = 17)
#plt.ylim(0, 110)
#plt.show()
