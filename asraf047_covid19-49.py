!pip install imutils

!pip install image-classifiers==1.0.0b1
import tensorflow as tf

import gc

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.applications import VGG16, DenseNet169

from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Activation, BatchNormalization, Dropout, LSTM, ConvLSTM2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input,Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation, LSTM, ConvLSTM2D, Lambda, Reshape, BatchNormalization, Bidirectional

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,TensorBoard,TerminateOnNaN, LearningRateScheduler

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

from tensorflow.keras.layers import Lambda, Reshape, DepthwiseConv2D, ZeroPadding2D, Add, MaxPooling2D,Activation, Flatten, Conv2D, Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers

from tensorflow.keras import backend as K



from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

from imutils import paths

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import random

import shutil

import cv2

import os

from datetime import datetime

%load_ext tensorboard
# dataset_path = './dataset'

# log_path = './logs'
# %%bash

# rm -rf dataset

# mkdir -p dataset/covid

# mkdir -p dataset/non-covid

# mkdir -p logs
# class_to_label_map = {'covid' : 1, 'non-covid' : 0}
# predY = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])
# cm_mat = confusion_matrix(testY, np.argmax(predY, axis = -1))

# cm_mat
# print(classification_report(testY, np.argmax(predY, axis = -1), target_names = ['covid', 'non-covid']))
!rm -rf dataset

!rm -rf logs
!pip install wget
import wget

url = "https://drive.google.com/uc?export=download&id=16uW6_hxgAm7y5SHQns2lrq7Ynz8BbBlw"

filename = wget.download(url, out="coronaVirusXRay.zip")

filename