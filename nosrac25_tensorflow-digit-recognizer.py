import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt



train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train_np = np.array(train, dtype='float32')

train_np.shape
train_x = np.array([x[1:] for x in train_np], dtype='float32')/255.0

train_y = np.array([y[0] for y in train_np], dtype='int32')



test_x = np.array(test, dtype='float32')/255.0
model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(784, activation='relu'),

    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dropout(0.15),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
model.fit(train_x, train_y, epochs=25)
predictions = model.predict(test_x)

test_y = [np.argmax(y) for y in predictions]



# the print is obnoxious

#print("ImageId,Label")

#for x in range(0, len(test_y)):

    #print(str(x+1) + "," + str(test_y[x]))