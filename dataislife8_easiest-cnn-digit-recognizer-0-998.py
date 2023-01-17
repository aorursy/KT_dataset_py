import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow import keras



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
ytrain = train['label']

xtrain = train.drop('label', axis=1)
xtrain = xtrain/255

test= test/255
xtrain.shape
xtrain = xtrain.values.reshape(42000,28,28,1)

test = test.values.reshape(28000,28,28,1)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
model.fit(xtrain, ytrain, epochs=10, verbose=1)
y_test = model.predict(test)

# Taking the highest probability output by the softmax activation function and labelling acoordingly

y_test = [np.argmax(y_test[i]) for i in range(y_test.shape[0])]



print('Sample of y_test :', y_test[:5])
results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_easy.csv",index=False)