# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (15,5)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/used-cars-price-prediction/train-data.csv',sep=',')
data
data.isnull().sum()
data = data.drop(['New_Price'], axis=1)
data = data.dropna(how='any',axis = 0)
#data.drop(data[data['Kilometers_Driven'] >= 6500000].index, axis=0, inplace=True)
data = data[data['Kilometers_Driven'] < 200000]
#data.to_numeric(data['Mileage'])

#data['Mileage'] = data['Mileage'].apply(lambda x: float(str(x)[:-4]))
data['Name'] = data['Name'].apply(lambda x: str(x).split(" ")[0])
data['Mileage'] = data['Mileage'].apply(lambda x: str(x).split(" ")[0])
data['Engine'] = data['Engine'].apply(lambda x: str(x).split(" ")[0])
data['Power'] = data['Power'].apply(lambda x: str(x).split(" ")[0])
data = data[data['Power'] != 'null']
data = data[data['Mileage'] != 0]
data.dtypes

data[['Mileage','Engine','Power']] = data[['Mileage','Engine','Power']].apply(pd.to_numeric)
data.isnull().sum()
data[75:4420]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
OH1 = OneHotEncoder()
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()
le5 = LabelEncoder()
le6 = LabelEncoder()
data_col = data.columns

data[data_col[1]] = le1.fit_transform(data[data_col[1]])
data[data_col[2]] = le2.fit_transform(data[data_col[2]])
data[data_col[3]] = le3.fit_transform(data[data_col[3]])
data[data_col[5]] = le4.fit_transform(data[data_col[5]])
data[data_col[6]] = le5.fit_transform(data[data_col[6]])
data[data_col[7]] = le6.fit_transform(data[data_col[7]])
data = data.reset_index()
OH1.fit(data[[data_col[1]]])
SCR_new = OH1.transform(data[[data_col[1]]]).toarray()
SCR_new = pd.DataFrame(SCR_new)

SCR_new.columns = le1.classes_

data = pd.concat((data, SCR_new), axis=1)

import seaborn as sns
correlation_matrix = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,vmax=0.3, vmin=-0.3,linewidths=1)
plt.show()
Y = data[['Kilometers_Driven']].values
data = data.drop(['Unnamed: 0','index','Name','Price','Owner_Type','Seats','Kilometers_Driven'], axis=1)
X = data[:].values
#X = data[['Name', 'Location','Year','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats', 'Kilometers_Driven']].values
data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
sc1 = MinMaxScaler()
sc2 = MinMaxScaler()
normalX = sc1.fit_transform(X)
normalY = sc2.fit_transform(Y)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(normalX,normalY,test_size=0.3, random_state = 42)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, TimeDistributed, LSTM, ConvLSTM2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, MaxPooling1D
from keras.datasets import mnist
from keras.utils import np_utils
from keras import regularizers

NNinput = train_x.shape[1]
act = 'relu'
opt = 'Adam'
los = 'mean_squared_error'

model = Sequential()
model.add(Dense(128, activation = act, input_shape = [NNinput,]))
model.add(Dense(128, activation = act))
model.add(Dense(128, activation = act))
model.add(Dense(1, activation = act))
model.compile(optimizer = opt, loss = los, metrics = ['mse'])
#model.summary()
batch_size = 20
epoch = 10
history = model.fit(train_x, train_y, epochs = epoch, batch_size = batch_size, verbose = 1, validation_data=(test_x, test_y))
pred = model.predict(test_x)
pred1 = sc2.inverse_transform(pred)
test_y1 = sc2.inverse_transform(test_y)
1 - abs(1 - test_y1 / pred1).mean()
answer = np.concatenate((pred1[:20], test_y1[:20]), axis=1 )
answer = pd.DataFrame(answer)
answer[0] = answer[0].astype(int)
answer[1] = answer[1].astype(int)
answer['percent'] = (answer[0] / answer[1] * 100).astype(int)

answer
plt.rcParams["figure.figsize"] = (15,5)
plt.plot(pred1[: ,])
plt.plot(test_y1[: ,])
plt.show()