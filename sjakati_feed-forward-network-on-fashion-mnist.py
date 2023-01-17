# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import RMSprop

from keras.utils.np_utils import to_categorical

from sklearn.cross_validation import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
## get training data

training_data = pd.read_csv('../input/fashion-mnist_train.csv')

## get testing data

testing_data = pd.read_csv('../input/fashion-mnist_test.csv')
## create training/validation split on training data

X = np.array(training_data.iloc[:, 1:])

y = to_categorical(np.array(training_data.iloc[:, 0]))

X_train, X_validate, y_train, y_validate = train_test_split(X, y)
model = Sequential()

## first hidden layer is fully connected to the input layer

model.add(Dense(400, input_dim=784, activation='relu')); model.add(Dropout(0.4))

## second hidden layer

model.add(Dense(200, activation='relu')); model.add(Dropout(0.3))

## third hidden layer

model.add(Dense(300, activation='sigmoid'));model.add(Dropout(0.2))

## output layer

model.add(Dense(10, activation='softmax'))

## compile the model using RMSProp

model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 50

batch_size = 128

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(X_validate, y_validate, batch_size=32)

print("Network's test score [loss, accuracy]: {0}".format(score))
model2 = Sequential()

## first hidden layer is fully connected to the input layer

model2.add(Dense(400, input_dim=784, activation='relu')); model.add(Dropout(0.4))

## second hidden layer

model2.add(Dense(200, activation='relu')); model.add(Dropout(0.3))

## third hidden layer

model2.add(Dense(100, activation='relu'));model.add(Dropout(0.2))

## fourth hidden layer

model2.add(Dense(50, activation='relu'))

## output layer

model2.add(Dense(10, activation='softmax'))

## compile the model using RMSProp

model2.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 20

batch_size = 128

model2.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

score = model2.evaluate(X_validate, y_validate, batch_size=32)

print("Network's test score [loss, accuracy]: {0}".format(score))
model3 = Sequential()

## first hidden layer is fully connected to the input layer

model3.add(Dense(400, input_dim=784, activation='relu')); model.add(Dropout(0.4))

## second hidden layer

model3.add(Dense(200, activation='sigmoid')); model.add(Dropout(0.3))

## third hidden layer

model3.add(Dense(100, activation='sigmoid'));model.add(Dropout(0.2))

## fourth hidden layer

model3.add(Dense(50, activation='sigmoid'))

## output layer

model3.add(Dense(10, activation='softmax'))

## compile the model using RMSProp

model3.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 20

batch_size = 128

model3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

score = model3.evaluate(X_validate, y_validate, batch_size=32)

print("Network's test score [loss, accuracy]: {0}".format(score))
X_test = np.array(testing_data.iloc[:, 1:])

y_test = to_categorical(np.array(testing_data.iloc[:, 0]))
score = model.evaluate(X_test, y_test, batch_size=32)

print("Network one's test score [loss, accuracy]: {0}".format(score))
score3 = model3.evaluate(X_test, y_test, batch_size=32)

print("Network three's test score [loss, accuracy]: {0}".format(score3))