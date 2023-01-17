import tensorflow as tf                       # deep learning library

import numpy as np                            # for matrix operations

import pandas as pd

import matplotlib.pyplot as plt               # for visualization

%matplotlib inline
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
train_data.head()
train_data.label.value_counts()
from sklearn.model_selection import train_test_split
X = train_data.drop('label', axis = 1)

y = train_data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
model = tf.keras.Sequential([

    tf.keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')     # The input shape is 784. 

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=10)



model.evaluate(X_test, y_test)
model = tf.keras.Sequential([

    tf.keras.layers.Dense(10, input_shape=(784,), activation='relu'),

    tf.keras.layers.Dense(10, activation='sigmoid')

])



# Compiling the model

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



# Fit the model

model.fit(X_train, y_train, batch_size= 128,epochs=30)



model.evaluate(X_test, y_test)
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
test_data.head()
test_data.info()
pred = model.predict(test_data)
predictions = [np.argmax(item) for item in pred]
my_submission = pd.DataFrame({'ImageId': test_data.index + 1, 'label': predictions})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)