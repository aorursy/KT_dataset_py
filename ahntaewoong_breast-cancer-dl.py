from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
data = datasets.load_breast_cancer()
data.keys()
pd.Series(data.target).unique()
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale
X = scale.fit_transform(data.data)
X
Y = data.target
Y
x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size = 0.3)
import tensorflow as tf
from tensorflow import keras
Inputs = tf.keras.Input(shape = (30))
x1 = tf.keras.layers.Dense(90, activation='relu')(Inputs)
x2 = tf.keras.layers.Dense(60, activation='relu')(x1)
outputs = tf.keras.layers.Dense(2, activation = 'softmax')(x2)
model = tf.keras.Model(inputs = Inputs, outputs = outputs)
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=200, batch_size=20, validation_split=0.2)
model.evaluate(x_test, y_test, batch_size=20)
pred = model.predict(x_test)
y_test.dtype, type(y_test)
#가장 가능성 높은값으로 분류
y_test
y_pred_class = np.argmax(pred,axis=1)
y_pred_class
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_class))
hist.history.keys()
b=pd.DataFrame(hist.history)
b
b.loss.min()
b.loss.argmin()
b.loc[169]
import matplotlib.pyplot as plt
plt.plot(b['loss'])
plt.plot(b['val_loss'])
plt.xlabel('epochs')
plt.ylim=([0, 1.0])
plt.legend(['loss','val_loss'])
plt.show()
plt.plot(b['accuracy'])
plt.plot(b['val_accuracy'])
plt.xlabel('epochs')
plt.ylim=([0, 1.0])
plt.legend(['accuracy','val_accuracy'])
plt.show()
