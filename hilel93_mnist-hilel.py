# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.callbacks import EarlyStopping

from keras.utils import normalize

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/digit-recognizer/train.csv') #load train data

test_data  = pd.read_csv('../input/digit-recognizer/test.csv') #load test data
# separate label from the data

X_train = train_data.drop(labels = ['label'], axis=1) 

y_train = train_data['label']



print('label values distribution:', y_train.value_counts())
print('missing values in train data:', X_train.isnull().sum().sum() )

print ('missing values in test data:', test_data.isnull().sum().sum())
X_train = X_train / 255.0

X_test = test_data / 255.0
# Reshape image data

X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)

print('X_train shape:', X_train.shape)

print('X_test shape:', X_test.shape)
#One hot encoding labels

y_train= to_categorical(y_train, num_classes=10)
#split data into train and validation sets

Xtrain, Xval, ytrain, yval = train_test_split(X_train, y_train, test_size = 0.2, random_state=124)

model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation='relu',padding='Same', input_shape=(28,28,1)))

model.add(Conv2D(64, kernel_size = (3,3), activation='relu',padding='Same'))



model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(64, kernel_size = (3,3), activation='relu',padding='Same'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation = 'softmax'))

model.summary()
model.compile(optimizer= keras.optimizers.SGD(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)

fitting = model.fit(Xtrain, ytrain, epochs=20,validation_split = 0.2, callbacks=[early_stopping_monitor])
score = model.evaluate(Xval, yval, verbose=0)

print('validation loss:', score[0])

print('validation accuracy:', score[1])


y_hat = model.predict(Xval)

y_pred = np.argmax(y_hat, axis=1)

y_true = np.argmax(yval, axis=1)

cm = confusion_matrix(y_true, y_pred)

print(cm)
plt.plot(fitting.history['val_loss'], color='b')

plt.title('Validation loss')

plt.ylabel('Loss value')

plt.xlabel('Epoch')

plt.show()

plt.plot(fitting.history['val_acc'], color='r')

plt.title('Validation accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
predictions = model.predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)