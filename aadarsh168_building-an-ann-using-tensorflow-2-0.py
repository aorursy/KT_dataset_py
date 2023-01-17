import numpy as np

import pandas as pd

import tensorflow as tf
# Checking the version of TensorFlow

tf.__version__
train = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

train.head()
# Creating a list of variables

features = train.columns.tolist()

features.remove('label')
test = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')

test.head()
X_train = np.array(train[features])

y_train = np.array(train['label'])
X_test = np.array(test[features])

y_test = np.array(test['label'])
X_train = X_train / 255.0

X_test = X_test / 255.0
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units = 128,activation = 'relu',input_shape = (784,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(units = 10,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'])
model.summary()
model.fit(X_train,y_train, epochs = 10)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Accuracy : {}%'.format(test_acc*100))