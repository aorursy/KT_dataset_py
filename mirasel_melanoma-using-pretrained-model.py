from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization, Dense
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16,inception_v3,resnet
from mlxtend.plotting import plot_confusion_matrix
from keras.optimizers import Adam, SGD , RMSprop
from keras.models import Model,Sequential
from keras.utils import to_categorical
from matplotlib.pyplot import figure
from keras import optimizers
from keras import models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import random
import pickle
import time
import cv2
import os
%matplotlib inline
# Total 400 data 200 train and 100 test and 100 validation
with open("../input/melanoma/melanoma_latest.pickle", "rb") as f:
    (x_train1,x_val1,x_test1,y_train1,y_val1,y_test1)=pickle.load(f,encoding='latin1')

# Total 256 data 152 train and 52 test and 52 validation
with open("../input/melanoma/melanoma_latest_256data.pickle", "rb") as f:
    (x_train2,x_val2,x_test2,y_train2,y_val2,y_test2)=pickle.load(f,encoding='latin1')
def vgg():
    vgg = vgg16.VGG16(include_top=False,weights="imagenet",input_shape=(224,224,3),pooling='max')
    x = vgg.output
    x = BatchNormalization()(x)
    x = Flatten()(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=vgg.input, outputs=predictions)
    return model

def inception():
    inception = inception_v3.InceptionV3(include_top=False,weights="imagenet",input_shape=(224,224,3),pooling='max')
    x = inception.output
    x = BatchNormalization()(x)
    x = Flatten()(x)
    predictions = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=inception.input, outputs=predictions)
    return model
def plotresult(y1,y2,ylabel,title,legends):
    plt.plot(y1)
    plt.plot(y2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    # plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()


def performacecompare(history1,history2):
    plt.figure(figsize=(15, 22))

    plt.subplot(421)
    y1 = history1.history['loss']
    y2 = history1.history['val_loss']
    plotresult(y1,y2,'Loss','VGG16 train and validation loss',['Train','Validation'])

    plt.subplot(422)
    y1 = history1.history['accuracy']
    y2 = history1.history['val_accuracy']
    plotresult(y1,y2,'Accuracy','VGG16 train and validation accuracy',['Train','Validation'])

    plt.subplot(423)
    y1 = history2.history['loss']
    y2 = history2.history['val_loss']
    plotresult(y1,y2,'Loss','InceptionV3 train and validation loss',['Train','Validation'])

    plt.subplot(424)
    y1 = history2.history['accuracy']
    y2 = history2.history['val_accuracy']
    plotresult(y1,y2,'Accuracy','InceptionV3 train and validation accuracy',['Train','Validation'])

    plt.subplot(425)
    y1 = history1.history['loss']
    y2 = history2.history['loss']
    plotresult(y1,y2,'Loss','VGG16 and InceptionV3 train loss',['VGG16','InceptionV3'])

    plt.subplot(426)
    y1 = history1.history['accuracy']
    y2 = history2.history['accuracy']
    plotresult(y1,y2,'accuracy','VGG16 and InceptionV3 train accuracy',['VGG16','InceptionV3'])

    plt.subplot(427)
    y1 = history1.history['val_loss']
    y2 = history2.history['val_loss']
    plotresult(y1,y2,'Loss','VGG16 and InceptionV3 validation loss',['VGG16','InceptionV3'])

    plt.subplot(428)
    y1 = history1.history['val_accuracy']
    y2 = history2.history['val_accuracy']
    plotresult(y1,y2,'accuracy','VGG16 and InceptionV3 validation accuracy',['VGG16','InceptionV3'])

    plt.show()
def testandcm(model,x_test,y_test,modelname=""):
    predict = model.predict(x_test)
    predict = np.argmax(predict,axis=1)
    acuracy = accuracy_score(y_test,predict)
    f1score = f1_score(y_test,predict)
    print(modelname+'Test accuracy = {}% and F1-Score = {}'.format(round(acuracy*100.0,2),round(f1score,2)))

    cm = confusion_matrix(y_test,predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(modelname+'Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['benign', 'malignant']); ax.yaxis.set_ticklabels(['benign', 'malignant']);
vgg1 = vgg()
vgg1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg1.summary()
vgg1_hist = vgg1.fit(x_train1,to_categorical(y_train1), validation_data=(x_val1, to_categorical(y_val1)),verbose=2, epochs=100)
inception1 = inception()
inception1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
inception1.summary()
inception1_hist = inception1.fit(x_train1,to_categorical(y_train1), validation_data=(x_val1, to_categorical(y_val1)),verbose=2, epochs=100)
# train performance comparision vgg and inception
performacecompare(vgg1_hist,inception1_hist)
#VGG16 test for 100 test data
testandcm(vgg1,x_test1,y_test1,"VGG16 ")
#InceptionV3 test for 100 test data
testandcm(inception1,x_test1,y_test1,"InceptionV3 ")
vgg2 = vgg()
vgg2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg2.summary()
vgg2_hist = vgg2.fit(x_train2,to_categorical(y_train2), validation_data=(x_val2, to_categorical(y_val2)),verbose=2, epochs=100)
inception2 = inception()
inception2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
inception2.summary()
inception2_hist = inception2.fit(x_train2,to_categorical(y_train2), validation_data=(x_val2, to_categorical(y_val2)),verbose=2, epochs=100)
# train performance comparision vgg and inception
performacecompare(vgg2_hist,inception2_hist)
#VGG16 test for 52 test data
testandcm(vgg2,x_test2,y_test2,"VGG16 ")
#InceptionV3 test for 100 test data
testandcm(inception2,x_test2,y_test2,"InceptionV3 ")