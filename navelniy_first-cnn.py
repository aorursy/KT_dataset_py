import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
X_train = train.drop('label',axis=1)
y_train = train['label']
X_train.head(3)
test.head(3)
print(y_train.shape)
print(X_train.shape)
print(pd.isnull(train).any().any()) #Return the True if there is one missed at least
print(pd.isnull(test).any().any())
X_train = X_train.values.reshape(-1,28,28,1)
X_test = test.values.reshape(-1,28,28,1)
print(X_train.shape, X_test.shape)
X_train = X_train / 255.0
X_test = X_test / 255.0
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5)

predictions = model.predict(X_test)
predictions[0]
res = np.argmax(predictions)
plt.figure()
plt.imshow(X_test[0,:,:,0])

