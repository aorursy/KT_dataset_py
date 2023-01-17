import os

from keras.callbacks import EarlyStopping

import pandas as pd

import numpy as np

from keras.models import Sequential

from keras.layers import Dense
def read_data(path):

    read = pd.read_csv(path)

    return read

def split_data(data_set):

    x_t = data_set.drop(columns=['wage_per_hour'])

    y_t = data_set[['wage_per_hour']]

    return x_t, y_t
def build_deep_learning_model(seq_model, no_cols):

    seq_model.add(Dense(10, activation='relu', input_shape=(no_cols,)))

    seq_model.add(Dense(10, activation='relu'))

    seq_model.add(Dense(1))

    seq_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

#     seq_model.summary()

    return seq_model
data = read_data('../input/hourly_wages_data.csv')

x_train, y_train = split_data(data)

no_columns = x_train.shape[1]

sequential_model = Sequential()

deep_learning_model = build_deep_learning_model(sequential_model, no_columns)

early_stopping_monitor = EarlyStopping(patience=3)

deep_learning_model.fit(x_train, y_train, validation_split=0.2, epochs=10, callbacks=[early_stopping_monitor])

#test model

print('are you femal?(0|1):')

isFemal =int(input())

print('are you married?(0|1):')

isMarried =int(input())

print('your age:')

age =int(input())

print('experience years:')

experienceYears =int(input())

print('education years:')

educationYears =int(input())

print('are you union?(0|1)')

union =int(input())

print('are you south?(0|1)')

south =int(input())

print('are you manufacturing?(0|1)')

manufacturing =int(input())

print('are you construction?(0|1)')

construction =int(input())

testData =np.array(([union, educationYears, experienceYears, age, isFemal, isMarried,south,manufacturing ,construction ])).reshape(1,no_columns)

pred = deep_learning_model.predict(testData)

# # print(pred)

print('your wage per hour: '+str(pred[0][0]))