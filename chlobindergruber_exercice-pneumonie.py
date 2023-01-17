# Import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Graphs

%matplotlib inline

from matplotlib import pyplot as plt

import seaborn as sns



# Sklearn

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn import datasets



# Keras

from keras.datasets import mnist

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.utils.np_utils import to_categorical



# Read in directories

import cv2 # opencv : OpenCV est une librairie opensource qui permet de manipuler des images en python.

import os # accès à des instuctions sur l'OS pour parcourir les répertoires notamment

import glob # permet de filtrer des fichiers dans les répertoires

import gc



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def lire_images(img_dir, xdim, ydim, nmax=5000) :

    """ 

    Lit les images dans les sous répertoires de img_dir

    nmax images lues dans chaque répertoire au maximum

    Renvoie :

    X : liste des images lues, matrices xdim*ydim

    y : liste des labels numériques

    label : nombre de labels

    label_names : liste des noms des répertoires lus

    """

    label = 0

    label_names = []

    X = []

    y=[]

    for dirname in os.listdir(img_dir):

        print(dirname)

        label_names.append(dirname)

        data_path = os.path.join(img_dir + "/" + dirname,'*g')

        files = glob.glob(data_path)

        n=0

        for f1 in files:

            if n>nmax : break

            img = cv2.imread(f1)

#           img = cv2.imread(f1, 1)   # Source : https://docs.opencv.org/master/db/deb/tutorial_display_image.html

# Je me suis aussi demandée si on pouvait ajouter un attribut du type : cv2.imread_grayscale(f1) pour modifier le 3 en 1 # Source : https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gga61d9b0126a3e57d9277ac48327799c80af660544735200cbe942eea09232eb822

            img = cv2.resize(img, (xdim,ydim))

#            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Source : https://stackoverflow.com/questions/60050816/converting-rgb-to-grayscale-python

            img.flatten()

            X.append(np.array(img))

            y.append(label)

            n=n+1

        print(n,' images lues')

        label = label+1

    X = np.array(X)

    y = np.array(y)

    gc.collect()

    return X,y, label, label_names
X,y,nlabels,labels = lire_images("../input/chest-xray-pneumonia/chest_xray/train/", 200, 200, 3000)
X.shape

# On voit bien 3 alors qu'on devrait avoir un 1
num_classes = 2

plt.imshow(X[0])

plt.title(labels[y[0]])
import random

plt.figure(figsize=(10,20))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    j = random.randint(0,len(X))

    plt.axis('off')

    plt.imshow(X[j])

    plt.title(labels[y[j]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train.shape
# Réseau convolutionnel simple

model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3), activation='relu'))

# Aussi testé : #model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 1), activation='relu'))

# Aussi testé : #model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 0), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

#model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



# Compilation du modèle

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
train = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=1)
# Test

scores = model.evaluate(X_test, y_test, verbose=0)

print("Score : %.2f%%" % (scores[1]*100))