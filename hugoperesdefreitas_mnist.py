# %tensorflow_version 2.x

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping 

from tensorflow.keras.utils import to_categorical
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



df = [train, test]
train
train.info()
samples = 20



print(train.values[0].shape)



nrows = 2

ncols = 10



for i in range(samples):

    img = train.iloc[i, 1:].values

    shape = int(np.sqrt(img.shape))

    img = img.reshape(shape, shape)

    plt.subplot(nrows, ncols, i + 1) 

    plt.imshow(img)



plt.show()

    
X = train.iloc[:, 1:].values

y = train.iloc[:, 0].values

X_pred = test.iloc[:, :].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.8)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

print(X_pred.shape)
y_train = to_categorical(y_train,  10)

y_test = to_categorical(y_test,  10)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
X_train = X_train.reshape(-1, shape, shape, 1)

X_test = X_test.reshape(-1, shape, shape, 1)
X_pred = scaler.transform(X_pred)

X_pred = X_pred.reshape(-1, shape, shape, 1)
X_train.shape
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',

                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

#model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

#model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

#model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=10, batch_size=5, validation_split=0.2, callbacks=[EarlyStopping(patience=2)] )
loss, acc = model.evaluate(X_test, y_test)

print('\n')
predictions = model.predict_classes(X_test)
new_pred = model.predict_classes(X_pred)
d = pd.DataFrame(new_pred, columns=['Label'])

d.index += 1

d.index.name = 'ImageId'

sns.countplot(x='Label', data=d)

d.to_csv('Predictions_2.csv')



d