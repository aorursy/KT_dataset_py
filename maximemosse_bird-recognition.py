# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import datasets

from keras.datasets import mnist

from keras import losses, regularizers

from keras.models import Sequential, load_model, Model

from keras.layers import Activation,Dense, Dropout, Flatten, BatchNormalization

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.constraints import maxnorm

from keras.utils.np_utils import to_categorical
import cv2
import glob
bird_dir="../input/100-bird-species/"
train_dir=bird_dir+"train/"
valid_dir=bird_dir+"valid/"
test_dir=bird_dir+"test/"
classes=[]
for dirname, _, filenames in os.walk(train_dir):
    folderName = os.path.basename(dirname)
    if folderName!="" :
        classes.append(folderName)
classes=sorted(classes)
X_train=[]
y_train=[]
X_test=[]
y_test=[]
X_valid=[]
y_valid=[]
for i in range(len(classes)):
    cur_train_dir=train_dir+classes[i]
    data_train_path=os.path.join(cur_train_dir,'*g')
    train_files=glob.glob(data_train_path)
    for f1 in train_files:
        X_train.append(cv2.resize(cv2.imread(f1),(100,100)))
        y_train.append(i)
    cur_test_dir=test_dir+classes[i]
    data_test_path=os.path.join(cur_test_dir,'*g')
    test_files=glob.glob(data_test_path)
    for f1 in test_files:
        X_test.append(cv2.resize(cv2.imread(f1),(100,100)))
        y_test.append(i)
    cur_valid_dir=valid_dir+classes[i]
    data_valid_path=os.path.join(cur_valid_dir,'*g')
    valid_files=glob.glob(data_valid_path)
    for f1 in valid_files:
        X_valid.append(cv2.resize(cv2.imread(f1),(100,100)))
        y_valid.append(i)
X_test=np.array(X_test, dtype=np.uint8)
X_train=np.array(X_train, dtype=np.uint8)
X_valid=np.array(X_valid, dtype=np.uint8)
X_train.shape
y_train_cat=to_categorical(y_train)
y_test_cat=to_categorical(y_test)
y_valid_cat=to_categorical(y_valid)
def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train=model.fit(X_train, y_train_cat, validation_data=(X_valid, y_valid_cat), epochs=20, batch_size=200, verbose=1)
plot_scores(train)
# Modèle CNN plus profond
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train=model.fit(X_train, y_train_cat, validation_data=(X_valid, y_valid_cat), epochs=20, batch_size=200, verbose=1)
model.evaluate(X_test, y_test_cat)
plot_scores(train)
train=model.fit(X_train, y_train_cat, validation_data=(X_valid, y_valid_cat), epochs=10, batch_size=200, verbose=1)
model.evaluate(X_test, y_test_cat)
plot_scores(train)
# Modèle CNN plus profond
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(20, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train=model.fit(X_train, y_train_cat, validation_data=(X_valid, y_valid_cat), epochs=20, batch_size=200, verbose=1)
plot_scores(train)
train=model.fit(X_train, y_train_cat, validation_data=(X_valid, y_valid_cat), epochs=20, batch_size=200, verbose=1)
model.evaluate(X_test,y_test_cat)
plot_scores(train)
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Classification Report')
print(classification_report(y_test, y_pred, target_names=classes))