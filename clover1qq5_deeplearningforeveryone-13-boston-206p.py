# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
import tensorflow as tf
seed = 0

np.random.seed(seed)

tf.random.set_seed(seed)
df = pd.read_csv('../input/housingcsv/housing.csv', delim_whitespace=True, header = None)
dataset = df.values

X = dataset[:, 0:13]

Y = dataset[:, 13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed)
model = Sequential()

model.add(Dense(30, input_dim=13, activation='relu'))

model.add(Dense(6, activation = 'relu'))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, epochs=200, batch_size=10)
Y_prediction = model.predict(X_test).flatten()

for i in range(10):

    label = Y_test[i]

    prediction = Y_prediction[i]

    print("실제가격: {:.3f}, 예상가격: {:.3f}". format(label, prediction))