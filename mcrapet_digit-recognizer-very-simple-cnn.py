# Python 3 environment

import numpy as np

import pandas as pd



train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



print(train.shape)

print(test.shape)
train.head()
Y_train = train['label']

X_train = train.drop(labels = ['label'], axis = 1)



# Is this faster?

#X = train.iloc[:,1:]

#y = train.iloc[:,0]
# Gray pixels are 0..255, make it 0..1.0

X_train = X_train / 255.0

test = test / 255.0



# Get rid of pandas DataFrame & reshape for CNN input

X_train = X_train.values.reshape((X_train.shape[0], 28, 28, 1))

X_test = test.values.reshape((test.shape[0], 28, 28, 1))
from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D



model1 = Sequential()

model1.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=(28, 28, 1), padding='same'))

model1.add(Conv2D(64, activation='relu', kernel_size=5, padding='same'))

model1.add(MaxPooling2D(2, 2))

model1.add(Dropout(0.4))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))

model1.add(Dense(10, activation='softmax'))

model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.fit(X_train, Y_train, batch_size=32, epochs=8, validation_split=0.2)
Y_test = model1.predict_classes(X_test)

df = pd.Series(Y_test, name='Label')

df.index += 1

df.to_csv("digit_recognizer-v5.csv", header=True, index_label='ImageId')