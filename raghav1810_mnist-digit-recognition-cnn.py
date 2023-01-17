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
import tensorflow as tf
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.layers import Dense, Conv2D, Dropout, Lambda, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
X_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')
y_train = X_train.iloc[:, 0].values.astype('float32')
X_train = X_train.iloc[:, 1:].values.astype('float32')

X_test = X_test.values.astype('float32')
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28 ,28, 1))
y_train = to_categorical(y_train, 10)
plt.imshow(X_train[427].reshape(28, 28), cmap='gray')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.16, random_state=42)
model1 = Sequential()

model1.add(Lambda(lambda x: x/255
                 , input_shape=(28,28,1)))

model1.add(Conv2D(8, kernel_size=(3, 3)
                 , activation='relu'))
model1.add(Dropout(0.5))
# model.add(BatchNormalization(axis = 3))

model1.add(Conv2D(8, kernel_size=(5, 5)
                , activation='relu'
                , padding='same'
                , strides=(2,2)))

model1.add(Conv2D(16, kernel_size=(7, 7)
                , activation='relu'
                , padding='same'))

model1.add(Flatten())
model1.add(Dropout(0.5))
model1.add(Dense(10, activation='softmax'))

model1.compile(optimizer='adam'
             , loss='categorical_crossentropy'
             , metrics=['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1.2, 
                                            factor=0.5, 
                                            min_lr=0.00001)
history = model1.fit(X_train, y_train, batch_size = 64, epochs = 52, 
         validation_data = (X_val, y_val), verbose = 1
                   , callbacks=[learning_rate_reduction])
preds = model1.evaluate(X_val, y_val)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1])) #model 1 deeper
predictions = model1.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("digitrecognitionKerasCNN.csv", index=False, header=True)
