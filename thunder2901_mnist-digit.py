import pandas as pd

import numpy as np

import keras 

from keras.models import Model

from keras.layers import *

from keras import optimizers

train = pd.read_csv("../input/digit-recognizer/train.csv")
train.head()
label = train['label']

feature = train.drop(['label'],axis='columns')

print(feature.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=200)



X_train.shape
y_train = keras.utils.to_categorical(y_train, 10)

y_test = keras.utils.to_categorical(y_test, 10)
print(y_train[0])
X_train /= 255

X_test /= 255
model = keras.Sequential()

model.add(Dense(512, activation='relu', input_shape=(784,)))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
history = model.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=12)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

epochs = len(acc)



fig = plt.figure(figsize=(10,6))

plt.plot(val_acc,c='red')

plt.plot(acc,c='blue')
X = pd.concat((X_train,X_test),axis=0)

X.shape
y = np.concatenate((y_train,y_test))

y.shape
model.fit(X,y,epochs=12)
df_test = pd.read_csv("../input/digit-recognizer/test.csv")

df_test.head()
testX = df_test.iloc[:, 0:784]

print(testX.shape)
testX = testX/255.0
pred = pd.DataFrame(model.predict(testX, batch_size=200))

pred = pd.DataFrame(pred.idxmax(axis=1))

pred.index.name = 'ImageId'

pred = pred.rename(columns = {0 : 'Label'}).reset_index()

pred['ImageId'] += 1

pred.head()
pred.to_csv('submission.csv',index=False)