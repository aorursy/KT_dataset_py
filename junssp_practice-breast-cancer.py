from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
data = datasets.load_breast_cancer()
data.keys()
scale = StandardScaler()
X = scale.fit_transform(data.data)
Y = data.target
X.shape
x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size = 0.3)
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Input(shape=x_train.shape[1]))
model.add(layers.Dense(300, activation='relu'))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
history = model.fit(x_train, y_train, epochs=30)
model.evaluate(x_test, y_test, batch_size=20)
prediction = model.predict(x_test)
prediction
for i, j in enumerate(prediction):
    if j > 0.5:
        prediction[i] = 1
    else:
        prediction[i] = 0
        
result_matrix = pd.DataFrame(confusion_matrix(y_test, prediction), index=['Real_0', 'Real_1'], columns=['Pred_0', 'Pred_1'])
result_matrix
print(classification_report(y_test, prediction))
history.history.keys()
history.history['loss']

plt.plot(history.history['loss'])
plt.plot(history.history['binary_accuracy'])
plt.xlabel('epochs')
plt.ylim=([0, 1.0])
plt.legend(['loss','binary_accuracy'])
plt.show()
