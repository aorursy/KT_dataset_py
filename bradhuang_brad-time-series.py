!pip install flaky

!pip install -U tensorflow-gpu

!pip install keras
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("/kaggle/input/timeseries/Train.csv")

df_test = pd.read_csv("/kaggle/input/timeseries/Test.csv")

df_train.head(5)
df_train.info()
df_test.info()
df_train['Datetime2'] = pd.to_datetime(df_train['Datetime'], format='%d-%m-%Y %H:%M')

df_train.head(5)
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



plt.figure(figsize=(50, 6))

plt.plot(df_train['Datetime2'],df_train['Count'])

plt.show()
from numpy import array

from keras.preprocessing.text import one_hot

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers.core import Activation, Dropout, Dense

from keras.layers import Flatten, LSTM

from keras.layers import GlobalMaxPooling1D

from keras.models import Model

from keras.layers.embeddings import Embedding

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer

from keras.layers import Input

from keras.layers.merge import Concatenate

from keras.layers import Bidirectional



import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt
X = list()

X = [x+1 for x in range(df_train.shape[0])]



print(X)
X = array(X).reshape(len(X), 1, 1)
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range = (0, 1))
Y = sc.fit_transform(df_train[['Count']].values)
model = Sequential()

model.add(LSTM(50, activation='relu', input_shape=(1, 1)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())
model.fit(X, Y, epochs=20, validation_split=0.2, batch_size=5)
test = list()

test = [x+df_train.shape[0]+1 for x in range(df_test.shape[0])]
test = array(test).reshape(len(test), 1, 1)
predicted = model.predict(test)
df_test_submit = pd.DataFrame({"Datetime": df_test['Datetime'], "Count":predicted.reshape(5112)})
df_test_submit.to_csv("submit.csv",index=False)