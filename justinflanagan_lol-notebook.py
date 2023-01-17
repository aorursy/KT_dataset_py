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
lol_data = pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv", index_col = "gameId")
lol_data.head()
index = lol_data.index.values
print(index[0:10])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = scaler.fit_transform(lol_data)
inverse = scaler.inverse_transform(normalized)
lol_data_normalized = pd.DataFrame(data= normalized)
lol_data_normalized = lol_data_normalized.set_index(index)

lol_data_normalized.columns = lol_data.columns

lol_data_normalized.head()

import seaborn as sns
sns.barplot(lol_data['redKills'],lol_data['blueWins'])
blue_data = lol_data.loc[:, :"blueGoldPerMin"]
blue_data.head()
red_data = lol_data.loc[:, "redWardsPlaced":]
red_data.head()
lol_train_data = lol_data_normalized.iloc[:7789, :]
lol_test_data = lol_data_normalized.iloc[7789:, :]
lol_test_data.shape
y = lol_train_data.loc[:, "blueWins"]
y.head()
X = lol_train_data.loc[:, "blueWardsPlaced" :]
X.head()
import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))

sns.heatmap(lol_data.corr())
kd_ratio = lol_data[["blueKills", "blueDeaths"]]
kd_ratio.head()

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
model = Sequential()
model.add(Dense(38, activation = "relu", input_shape = (38,)))
model.add(Dropout(.2))
model.add(Dense(86, activation = "relu"))
model.add(Dropout(.2))
model.add(Dense(86, activation = 'relu'))
model.add(Dense(2, activation = "softmax"))



model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = "adam",
              metrics = ['accuracy'])
model.fit(X,y,
          epochs = 100,
         batch_size = 1)