# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
DATA_DIR = '../input/international-airline-passengers/international-airline-passengers.csv'

data = pd.read_csv(DATA_DIR)

data
# Drop last row

data = data[:-1]
# Rename columns

data.rename(columns={'Month': 'date', 'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60': 'total_passangers'}, inplace=True)

data = data.drop('date', axis=1)
data.shape
# split a univariate sequence into samples

def split_sequence(sequence, n_steps):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps

        # check if we are beyond the sequence

        if end_ix > len(sequence)-1:

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)
# Split data to train and test subsets

def train_test_split(data, n_test):

    return data[:-n_test], data[-n_test:]
X, y = split_sequence(data['total_passangers'], 6)

del(data)
X = X.reshape((-1, 6, 1))
X_train, X_test = train_test_split(X, 12)

print(f'{X_train.shape} {X_test.shape}')
y_train, y_test = train_test_split(y, 12)

print(f'{y_train.shape} {y_test.shape}')
model = Sequential()



model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=X_train.shape[1:]))

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(50, activation='relu'))

model.add(Dense(1))



model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10000, validation_data=[X_test, y_test], verbose=0)
predictions = model.predict(X_test)



for i in range(len(X_test)):

    print(y_test[i], predictions[i][0], abs(y_test[i]-predictions[i][0]))
def calculate_mse(preds, labels):

    amount = len(preds)

    mse_sum = 0

    for (pred, label) in zip(preds, labels):

        mse_sum += abs(pred[0]-label) 

    mse_sum /= amount

    return mse_sum
predictions = model.predict(X_train)

calculate_mse(predictions, y_train)
predictions = model.predict(X_test)

calculate_mse(predictions, y_test)