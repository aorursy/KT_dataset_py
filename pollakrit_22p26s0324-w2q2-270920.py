import tensorflow as tf

import keras



import sklearn

from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import jaccard_score



from scipy import stats



import seaborn as sns



import skimage

from skimage.transform import rotate



from tensorflow.keras.utils import to_categorical

from tensorflow.keras import Sequential

from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, UpSampling2D, GlobalMaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, concatenate

from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input

from tensorflow.keras.models import Model, load_model



from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications import NASNetMobile, Xception, DenseNet121, MobileNetV2, InceptionV3, InceptionResNetV2, vgg16, resnet50, inception_v3, xception, DenseNet201

from tensorflow.keras.applications.vgg16 import VGG16





from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import CSVLogger

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from datetime import datetime



import numpy as np

import os

import cv2

import pandas as pd

# import imutils

import random

from PIL import Image

import matplotlib.pyplot as plt
df = pd.read_csv('../input/titanic/gender_submission.csv')

df
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/train.csv')



df_train = df_train.dropna(subset=['Age']).drop(['Cabin', 'Embarked'], axis=1)

# df_test = df_test.dropna(subset=['Age']).drop(['Cabin', 'Embarked'], axis=1)



df_train
for num, i in enumerate(df_train['Name']):

    df_train['Name'][num] = i.split(',')[0]



for num, i in enumerate(df_test['Name']):

    df_test['Name'][num] = i.split(',')[0]

    

df_train
df_train['Sex'] = df_train['Sex'].replace(['male', 'female'], [1, 0])

df_test['Sex'] = df_test['Sex'].replace(['male', 'female'], [1, 0])

df_train
Names = np.unique(df_train['Name'])

Names
df_train['Name'] = df_train['Name'].replace(Names,range(len(Names)))

df_train
Tickets = np.unique(df_train['Ticket'])

Tickets
df_train['Ticket'] = df_train['Ticket'].replace(Tickets,range(len(Tickets)))

df_train
# Number of Name and Ticket are too much, so I will drop them!!!!!!!

df_train.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

df_test.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)



df_train = df_train.reset_index().drop(['index'], axis=1)

df_train
X = df_train.drop(['Survived'], axis=1)

X
Y = df_train['Survived']

Y
X = np.array(X)

Y = np.array(Y)

X[:5], Y[:5]
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report
kfold = KFold(n_splits=5, random_state=42)



print('===============================================================')

DT_model = DecisionTreeClassifier(criterion='entropy', max_depth=None)

for count, (train_index, valid_index) in enumerate(kfold.split(X)):

    DT_model = DT_model.fit(X[train_index], Y[train_index])

    y_pred = DT_model.predict(X[valid_index])

    y_train_score = DT_model.score(X[train_index], Y[train_index])

    y_valid_score = DT_model.score(X[valid_index], Y[valid_index])

    y_score = DT_model.score(X, Y)

    print('KFold :', count+1)

    print('Decision Tree Train Score :', y_train_score)

    print('Decision Tree Validation Score :', y_valid_score)

    print('Decision Tree Score :', y_score, '\n')

    print(classification_report(Y[valid_index], y_pred))

    print('===============================================================')

    

y_pred = DT_model.predict(X)

y_score = DT_model.score(X, Y)

print('Final Decision Tree Model')

print('Decision Tree Final Score :', y_score, '\n')

print(classification_report(Y, y_pred))

print('===============================================================')
from sklearn.naive_bayes import GaussianNB
print('===============================================================')

gnb_model = GaussianNB()

for count, (train_index, valid_index) in enumerate(kfold.split(X)):

    gnb_model.fit(X[train_index], Y[train_index])

    y_pred = gnb_model.predict(X[valid_index])

    y_train_score = gnb_model.score(X[train_index], Y[train_index])

    y_valid_score = gnb_model.score(X[valid_index], Y[valid_index])

    y_score = gnb_model.score(X, Y)

    print('KFold :', count+1)

    print('GaussianNB Train Score :', y_train_score)

    print('GaussianNB Validation Score :', y_valid_score)

    print('GaussianNB Score :', y_score, '\n')

    print(classification_report(Y[valid_index], y_pred))

    print('===============================================================')

    

y_pred = gnb_model.predict(X)

y_score = gnb_model.score(X, Y)

print('Final GaussianNB Model')

print('GaussianNB Final Score :', y_score, '\n')

print(classification_report(Y, y_pred))

print('===============================================================')
from sklearn.neural_network import MLPClassifier
print('===============================================================')

MLP_model = MLPClassifier(hidden_layer_sizes=(64,64), random_state=1)

for count, (train_index, valid_index) in enumerate(kfold.split(X)):

    MLP_model.fit(X[train_index], Y[train_index])

    y_pred = MLP_model.predict(X[valid_index])

    y_train_score = MLP_model.score(X[train_index], Y[train_index])

    y_valid_score = MLP_model.score(X[valid_index], Y[valid_index])

    y_score = MLP_model.score(X, Y)

    print('KFold :', count+1)

    print('MLPClassifier Train Score :', y_train_score)

    print('MLPClassifier Validation Score :', y_valid_score)

    print('MLPClassifier Score :', y_score, '\n')

    print(classification_report(Y[valid_index], y_pred))

    print('===============================================================')

    

y_pred = MLP_model.predict(X)

y_score = MLP_model.score(X, Y)

print('Final MLPClassifier Model')

print('MLPClassifier Final Score :', y_score, '\n')

print(classification_report(Y, y_pred))

print('===============================================================')