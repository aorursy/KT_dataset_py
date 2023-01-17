import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras

%matplotlib inline



train = pd.read_csv('../input/cars-train.csv')

val = pd.read_csv('../input/cars-test.csv')



x_train = pd.get_dummies(train.drop(['class', 'car.id'], axis=1))

x_val = pd.get_dummies(val.drop(['class', 'car.id'], axis=1))

y_train = pd.get_dummies(train['class'])

y_val = pd.get_dummies(val['class'])
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, PReLU, BatchNormalization



model = Sequential()

model.add(Dense(128, input_dim=(21), activation = 'selu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Dense(128, activation = 'selu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Dense(128, activation = 'selu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Dense(128, activation = 'selu'))

model.add(Dropout(0.2))

model.add(BatchNormalization())

model.add(Dense(4, activation = 'softmax'))



model.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 600, batch_size = 32)
score = model.evaluate(x_train, y_train)



print('\ntrain loss is: ' + str(score[0].round(4)))

print('train accuracy is: ' + str(score[1]))



score = model.evaluate(x_val, y_val)



print('\ntest loss is: ' + str(score[0].round(4)))

print('test accuracy is: ' + str(score[1]))
test = pd.read_csv('../input/cars-final-prediction.csv')

test_new = pd.get_dummies(test.drop(['car.id'], axis=1))

preds = model.predict_classes(test_new)

keras_dict = {0: 'acc', 1: 'good', 2: 'unacc', 3: 'vgood'}

converted_preds = []

for prediction in preds:

    converted_preds.append(keras_dict[prediction])

test['class']=converted_preds

output_csv = test[['car.id', 'class']]

output_csv
output_csv.to_csv('car-submission.csv', index=False)