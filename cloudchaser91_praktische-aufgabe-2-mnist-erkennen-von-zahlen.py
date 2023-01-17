# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical

# Input data files are available in the "../input/" directory.

# Load Train and Test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Trainings-Datensatz: (Zeilen, Spalten)")
print(train.shape)

print("Test-Datensatz: (Zeilen, Spalten)")
print(test.shape)
# separting 'label' column from train dataset
train_label = train["label"]

# dropping 'Label' column 
## axis = 1 gibt an, dass die gesamte Spalte von "label" genommen wird
## .values erstellt eine Numpy-Repräsentation der Daten
train = train.drop("label", axis = 1).values
# Normalizing the data
train = train/255.0
test= test/255.0
# label encoding of train_label dataset which has category of 0 to 9 values using one hot encoding
train_label = to_categorical(train_label)
# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(128, input_dim = 784))
model.add(Activation('relu'))
model.add(Dropout(0.10))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.10))

model.add(Dense(10))
model.add(Activation('softmax'))
# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print("Training...")
model.fit(train, train_label, epochs=20, batch_size=64, validation_split=0.2, verbose=1)
print("Generating test predictions...")
preds = model.predict_classes(test, verbose=1)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "keras-mlp.csv")
print("done")