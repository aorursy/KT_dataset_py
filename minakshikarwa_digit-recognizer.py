# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

#from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from sklearn.model_selection import train_test_split



from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping

from keras import models



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv')

train.shape, test.shape
train.head()
X = train.drop('label', 1)

Y = train['label']

X.shape, Y.shape
X.head()
##Standaration of data 

X = X.astype('float32')

test = test.astype('float32')

X /=255

test /= 255

print('train shape:', X.shape)

print('train samples 0 :', X.shape[0])

print('test samples 0 :', test.shape[0])

print(Y.value_counts())

sns.countplot(Y)
# convertion of label vector to binary classes

num_classes = 10

train_label = keras.utils.to_categorical(Y, num_classes)

train_label
# train and val split of the data set

X_train, X_val, Y_train, Y_val = train_test_split(X, train_label, test_size=0.20)

X_train.shape, Y_train.shape, X_val.shape, Y_val.shape
# model building

model = Sequential()

model.add(Dense(256, input_dim=784, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(128, activation='tanh'))

model.add(Dropout(0.25))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer =keras.optimizers.Adam(),

              metrics= ['accuracy'])
es = EarlyStopping(monitor='accuracy', patience=5)

md = ModelCheckpoint(filepath="/content/best_model.h5", verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train,

              batch_size =128,

              epochs =10,

              verbose =1,

              validation_data =(X_val,Y_val))
pred = model.predict_classes(X_val)

pred
pd.crosstab(np.argmax(Y_val, axis =1), pred)
pred_test = model.predict_classes(test)

pred_test
Y_val
np.savetxt('pred_98.csv' , pred_test, header="Label")
# Plot loss trajectory throughout training.

plt.figure(1, figsize=(14,5))

plt.subplot(1,2,1)

plt.plot(history.history['loss'], label='train')

plt.plot(history.history['val_loss'], label='valid')

plt.xlabel('Epoch')

plt.ylabel('Cross-Entropy Loss')

plt.legend()



plt.subplot(1,2,2)

plt.plot(history.history['accuracy'], label='train')

plt.plot(history.history['val_accuracy'], label='valid')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()
score = model.evaluate(X_val, Y_val, verbose=0)

print('Test cross-entropy loss: %0.5f' % score[0])

print('Test accuracy: %0.2f' % score[1])
print(score)