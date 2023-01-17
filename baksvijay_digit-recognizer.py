# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import keras



from keras.datasets import mnist



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import seaborn as sns
# train = pd.read_csv("../input/train.csv")

(xtrain, ytrain), (xvalid, yvalid) = mnist.load_data()
# train = train.sample(frac=0.1)

print(xtrain.shape, ytrain.shape, xvalid.shape, yvalid.shape)
xtrain = xtrain.reshape(60000, 784).astype('float32') / 255

xvalid = xvalid.reshape(10000, 784).astype('float32') / 255

print(xtrain.shape, ytrain.shape, xvalid.shape, yvalid.shape)
ytrain = keras.utils.to_categorical(ytrain, num_classes=10)

yvalid = keras.utils.to_categorical(yvalid, num_classes=10)
# model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 32), random_state=1, max_iter=1000, early_stopping=True, verbose=2)

model = Sequential()



model.add(Dense(128, input_dim=784, activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='sgd',

              metrics=['accuracy'])
model.fit(xtrain, ytrain, validation_data=(xvalid, yvalid), verbose=1, epochs=20, batch_size=128)
score = model.evaluate(xvalid, yvalid, batch_size=128)
score
xvalid
pred = model.predict(xvalid)
# f1_score(valid_df['label'], pred, average='micro')
# print(classification_report(valid_df['label'], pred))
pred = pred.argmax(axis=1)
yvalid = yvalid.argmax(axis=1)
confusion_matrix(yvalid, pred)
sns.heatmap(confusion_matrix(yvalid, pred))
f1_score(yvalid, pred, average='micro')
image_index = 4444

plt.imshow(xvalid[image_index].reshape(28, 28),cmap='Greys', label=yvalid[image_index])

plt.legend()

plt.show()

yvalid[image_index]
image_index = 3333

plt.imshow(xvalid[image_index].reshape(28, 28),cmap='Greys', label=yvalid[image_index])

plt.legend()

plt.show()

yvalid[image_index]