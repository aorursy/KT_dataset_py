# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
# Import EarlyStopping
from keras.callbacks import EarlyStopping
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y_train = np.array(train["label"])

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

X_train.describe()
test.describe()
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
# Save the number of columns: n_cols
n_cols = X_train.shape[1]

# Splitting training data
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=12)
# Set up the model: model
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dropout(0.15))
model.add(Dense(62, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(35, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=4)

hist=model.fit(X_train,Y_train,epochs=30,validation_data=(X_val,Y_val),callbacks=[early_stopping_monitor], 
               shuffle=True)
#print(hist.history['val_loss'])

Y_val_pred=np.argmax(model.predict(X_val), axis = 1)
Y_val=np.argmax(Y_val, axis=1)
print(Y_val_pred[:])
print(Y_val)
print(Y_val_pred.shape)
print(Y_val.shape)
confusion_matrix(Y_val, Y_val_pred)
# predict results
results = model.predict(test)

# select the index with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("mnist_result.csv",index=False)
