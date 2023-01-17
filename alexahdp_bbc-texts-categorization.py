import os

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import keras

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer, LabelEncoder

%matplotlib inline



models = keras.models

layers = keras.layers
data = pd.read_csv('../input/bbc-text.csv')
data.head()
data['category'].value_counts()
max_words=1000

tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words, char_level=False)
x_train, x_test, y_train, y_test = train_test_split(

    data['text'],

    data['category'],

    test_size=0.2,

    random_state=42

)
tokenize.fit_on_texts(x_train)

x_train = tokenize.texts_to_matrix(x_train)

x_test = tokenize.texts_to_matrix(x_test)
encoder = LabelEncoder()

encoder.fit(data['category'])

y_train = encoder.transform(y_train)

y_test = encoder.transform(y_test)
num_classes = int(np.max(y_train) + 1)
y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)
batch_size = 32

epochs = 2

drop_ratio = 0.5
[max_words, num_classes]
model = models.Sequential()

model.add(layers.Dense(512, input_shape=(max_words,)))

model.add(layers.Activation('relu'))

model.add(layers.Dense(num_classes))

model.add(layers.Activation('softmax'))
model.compile(

    loss='categorical_crossentropy',

    optimizer='adam',

    metrics=['accuracy']

)
x_train.shape
history = model.fit(

    x_train,

    y_train,

    batch_size=batch_size,

    epochs=epochs,

    verbose=1,

    validation_split=0.1

)
score = model.evaluate(

    x_test,

    y_test,

    batch_size=batch_size,

    verbose=1

)
print('Test loss: ', score[0])

print('Test accuracy: ', score[1])