import tensorflow as tf

print(tf.__version__)
tf.random.set_seed(42)
import pandas as pd

from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state =42)
from keras.utils import to_categorical
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
# Import `Sequential` from `keras.models`

from tensorflow.keras import layers
# Initialize the constructor

model = tf.keras.Sequential()
# Add an input layer 

model.add(layers.Dense(3, activation='sigmoid',input_shape=(4,)))



# Add an output layer 

model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='SGD',

              metrics=['accuracy'])
print(len(model.layers))

print(len(model.inputs))

print(len(model.outputs))
model.summary()
model.get_config()
print(X_train.shape)
y_train.shape
from keras.utils import to_categorical

model.fit(X_train, y_train,epochs=100, batch_size=1, verbose=1)
import numpy as np

y_pred = np.round(model.predict(X_test))

y_pred[0:10]