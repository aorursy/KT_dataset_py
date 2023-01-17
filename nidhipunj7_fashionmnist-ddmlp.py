# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout

from sklearn.metrics import confusion_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"

data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)
df_train.head()
# Read data



labels = df_train.ix[:,0].values.astype('int32')

X_train = (df_train.ix[:,1:].values).astype('float32')



X_test = (df_test.ix[:,1:].values).astype('float32')



labels_test = (df_test.ix[:,0].values).astype('float32')



# convert list of labels to binary class matrix

y_train = np_utils.to_categorical(labels) 



y_test =  np_utils.to_categorical(labels_test)
# pre-processing: divide by max and substract mean

scale = np.max(X_train)

X_train = X_train/scale

X_test = X_test/scale



mean = np.std(X_train)

X_train =X_train - mean

X_test =X_test - mean
input_dim = X_train.shape[1]

nclass = y_train.shape[1]
# Here's a Deep Dumb MLP (DDMLP)

model = Sequential()

model.add(Dense(128, input_dim=input_dim))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(nclass))

model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
print("Training...")

model.fit(X_train, y_train, nb_epoch=10, batch_size=16, validation_split=0.1, verbose=2)
preds = model.predict_classes(X_test, verbose=0)
import seaborn as sns

import matplotlib.pyplot as plt

LABELS = ["T-shirt/top",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandal",

    "Shirt",

    "Sneaker",

    "Bag",

    "Ankle boot"]

# pred_y = [1 if e > 0.034197 else 0 for e in error_df_test.Reconstruction_error.values]

conf_matrix = confusion_matrix(labels_test, preds)

plt.figure(figsize=(12, 12))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()