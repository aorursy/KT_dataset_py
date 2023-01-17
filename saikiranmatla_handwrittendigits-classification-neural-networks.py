#Importing Packages
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Checking Tensorflow version
print(tf.__version__)
#Loading data from keras API
(X_train, y_train ), (X_test, y_test) = keras.datasets.mnist.load_data()
#Length of train values
len(X_train)
len(X_test)
#Shape of X_train
X_train.shape
X_train[0]
#Viewing the values in train
plt.matshow(X_train[2])
X_train.shape
#Scaling the values
X_train = X_train/255
X_test = X_test/255
#Flattening to single layer stack
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
print(X_test_flattened.shape)
X_train_flattened[0]
#Creating a simple model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape = (784,), activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']

)

model.fit(X_train_flattened, y_train, epochs = 10)
#Evaluating the model performance using test values
model.evaluate(X_test_flattened, y_test)
plt.matshow(X_test[1])
#Predicting the values
y_pred = model.predict(X_test_flattened)
y_pred[1]
np.argmax(y_pred[1])
#Printing indices of array
y_pred_labels = [np.argmax(i) for i in y_pred]
print(y_pred_labels[:5])
#confusion matrix
cm = tf.math.confusion_matrix(labels= y_test, predictions= y_pred_labels)
cm
#creating a model with one hidden layer 
model = keras.Sequential([
    keras.layers.Dense(100, input_shape = (784,), activation = 'relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']

)

model.fit(X_train_flattened, y_train, epochs = 10)
model.evaluate(X_test_flattened, y_test)
cm = tf.math.confusion_matrix(labels= y_test, predictions= y_pred_labels)
cm
#Creating a model 
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']

)

model.fit(X_train, y_train, epochs = 10)