# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Directive pour afficher les graphiques dans Jupyter
%matplotlib inline

# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
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
import os
import glob
import sys
classes=[]
for dirname, _, filenames in os.walk('/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/train/'):
    folderName = os.path.basename(dirname)
    if folderName!="" :
        classes.append(folderName)
classes=sorted(classes)
X_train=[]
y_train=[]
X_test=[]
y_test=[]
for i in range(len(classes)):
    train_dir='/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/train/'+classes[i]
    data_train_path=os.path.join(train_dir,'*g')
    train_files=glob.glob(data_train_path)
    for f1 in train_files:
        X_train.append(np.array(cv2.resize(cv2.imread(f1),(80,80))))
        y_train.append(i)
    test_dir='/kaggle/input/stanford-car-dataset-by-classes-folder/car_data/car_data/test/'+classes[i]
    data_test_path=os.path.join(test_dir,'*g')
    test_files=glob.glob(data_test_path)
    for f1 in test_files:
        X_test.append(np.array(cv2.resize(cv2.imread(f1),(80,80))))
        y_test.append(i)
print(len(X_train))
print(len(X_test))
plt.imshow(X_train[0])
plt.title(classes[y_train[0]])
np.array(X_train).shape
X_train=np.array(X_train)
X_test=np.array(X_test)
X_train=X_train/255
X_test=X_test/255
model = Sequential([
    Flatten(input_shape=(80, 80,3)),
    Dense(1024, activation='relu'),
    Dense(256, activation='relu'),
    Dense(len(classes))
])

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(80, 80, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation = 'softmax'))
model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=200, verbose=1)
def plot_scores(train) :
    accuracy = train.history['accuracy']
    val_accuracy = train.history['val_accuracy']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'b', label='Score apprentissage')
    plt.plot(epochs, val_accuracy, 'r', label='Score validation')
    plt.title('Scores')
    plt.legend()
    plt.show()
#plot_scores(train)
#train2 = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=40, batch_size=200, verbose=1)
#plot_scores(train2)
y_train1 = to_categorical(y_train)
y_test1 = to_categorical(y_test)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(80, 80, 3), activation='relu'))
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
train = model.fit(X_train, y_train1, validation_data=(X_test, y_test1), epochs=30, batch_size=200, verbose=1)
plot_scores(train)