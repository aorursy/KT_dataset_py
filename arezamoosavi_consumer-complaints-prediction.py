# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from __future__ import absolute_import

from __future__ import division

from __future__ import print_function
import itertools

import os



%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow as tf



from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from sklearn.metrics import confusion_matrix



from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.preprocessing import text, sequence

from keras import utils



# This code was tested with TensorFlow v1.4

print("You have TensorFlow version", tf.__version__)
df = pd.read_csv('../input/Consumer_Complaints.csv', encoding='latin-1')

df.head()
col = ['Issue', 'Product']

df = df[col]

df = df[pd.notnull(df['Issue'])]

df.head()
df.isnull().sum()
df['Product'].value_counts()
# Split data into train and test

train_size = int(len(df) * .8)

print ("Train size: %d" % train_size)

print ("Test size: %d" % (len(df) - train_size))
train_narrative = df['Issue'][:train_size]

train_product = df['Product'][:train_size]



test_narrative = df['Issue'][train_size:]

test_product = df['Product'][train_size:]
max_words = 1000

tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_narrative) # only fit on train

x_train = tokenize.texts_to_matrix(train_narrative)

x_test = tokenize.texts_to_matrix(test_narrative)
# Use sklearn utility to convert label strings to numbered index

encoder = LabelEncoder()

encoder.fit(train_product)

y_train = encoder.transform(train_product)

y_test = encoder.transform(test_product)
# Converts the labels to a one-hot representation

num_classes = np.max(y_train) + 1

y_train = utils.to_categorical(y_train, num_classes)

y_test = utils.to_categorical(y_test, num_classes)
# Inspect the dimenstions of our training and test data (this is helpful to debug)

print('x_train shape:', x_train.shape)

print('x_test shape:', x_test.shape)

print('y_train shape:', y_train.shape)

print('y_test shape:', y_test.shape)
batch_size = 32

epochs = 5
# Build the model

model = Sequential()

model.add(Dense(512, input_shape=(max_words,)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
history = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_split=0.1)
# Evaluate the accuracy of our trained model

score = model.evaluate(x_test, y_test,

                       batch_size=batch_size, verbose=1)

print('Test score:', score[0])

print('Test accuracy:', score[1])
# Here's how to generate a prediction on individual examples

text_labels = encoder.classes_ 



for i in range(10):

    prediction = model.predict(np.array([x_test[i]]))

    predicted_label = text_labels[np.argmax(prediction)]

    print(test_narrative.iloc[i][:50], "...")

    print('Actual label:' + test_product.iloc[i])

    print("Predicted label: " + predicted_label + "\n")