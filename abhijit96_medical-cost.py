# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/insurance.csv')

cols = ['sex', 'smoker', 'region']

for c in cols:

    le = LabelEncoder()

    data[c] = le.fit_transform(data[c])
target = data.charges.values

data.drop('charges', axis=1, inplace=True)

data = data.values
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout
model = Sequential()

model.add(Dense(20, input_dim=6))

model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
from sklearn.model_selection import train_test_split

X, X_val, Y, Y_val = train_test_split(data, target, test_size = 0.07)
model.fit(X,Y, epochs=1000, validation_data = (X_val, Y_val), batch_size=10)
pred = model.predict(X_val).tolist()

actual = Y_val.tolist()

diff = [abs(pred[i][0] - actual[i]) for i in range(len(pred))]

for i in range(20):

    print('%f\t%f\t%f'%(pred[i][0], actual[i], diff[i]))