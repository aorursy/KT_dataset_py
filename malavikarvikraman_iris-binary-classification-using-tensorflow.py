# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



data = pd.read_csv('../input/iris/Iris.csv')
data.columns
data.shape
set(data['Species'])
import tensorflow as tf

from tensorflow import keras

from sklearn.preprocessing import OneHotEncoder
X=data.drop(['Id','Species'],axis=1)
X[:5]
y= data.Species

y= np.asanyarray(y).reshape(-1,1)

encoder = OneHotEncoder(sparse=False)

y = encoder.fit_transform(y)

y[:5]
X.shape
y.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

y_train  = y_train.astype(float)

y_test  = y_test.astype(float)

batch_size =len(X_train)

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

model = Sequential()



model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))

model.add(Dense(10, activation='relu', name='fc2'))

model.add(Dense(3, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)

model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
print('Neural Network Model Summary: ')

print(model.summary())
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=1)
print(history.history.keys())
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
testing=model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1]+' of Model on testing data', testing[1]*100))
from sklearn.metrics import classification_report, confusion_matrix  

predictions = model.predict(X_test[0:1])

print("Input features ",X_test.head(1))

print("Predicted value: ",np.argmax(predictions,axis=1))

print("Actual value:",np.argmax(y_test[0]))