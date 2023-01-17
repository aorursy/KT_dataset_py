import tensorflow as tf

from tensorflow import keras





import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



print(tf.__version__)
print(type(x_train))

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
print(np.max(x_train))

print(np.mean(x_train))
print(y_train)
class_names = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
plt.figure()

plt.imshow(x_train[1])

plt.colorbar()
x_train = x_train/255.0

x_test = x_test/255.0

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Flatten, Dense
model = Sequential()

model.add(Flatten(input_shape = (28,28) ))

model.add(Dense(128, activation= 'relu') )

model.add(Dense(10, activation = 'softmax'))
model.summary()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 10)
from sklearn.metrics import accuracy_score
y_pred = model.predict_classes(x_test)
print(y_pred)
accuracy_score(y_test, y_pred)
pred = model.predict(x_test)
print(pred)
print(pred[0])

print(y_pred[0])

print(np.argmax(pred[0]))