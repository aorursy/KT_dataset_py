# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
base = pd.read_csv('../input/kc_house_data.csv')
base.head()
X = base.iloc[:,[3,4,5,6,7,11,12,13,14,17,18]]
X.head()
y = base.iloc[:,2]
y.head()
X = X.values
y = y.values.reshape(-1,1)
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = \
train_test_split(X, y, test_size=.2, random_state=0)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
dic_loss = {}
lr_ = []
mae_ = []

lr_list = [.000001,.000005,.00001,.00005,.0001,.0005,.001,.005,.01,.05]
count = 0

for lr in lr_list:
    
    print('lr =',lr)
    count += 1
    print(str(count)+'/'+str(len(lr_list)))

    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    model = Sequential()
    model.add(Dense(6, input_dim=11, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=opt, metrics=['mse','mae'])
    history = model.fit(X_treinamento, y_treinamento, epochs=100, verbose=0, batch_size=25)
    
    previsoes = model.predict(X_teste)

    previsoes = scaler_y.inverse_transform(previsoes)
    #y_teste = scaler_y.inverse_transform(y_teste)

    mae = mean_absolute_error(scaler_y.inverse_transform(y_teste), previsoes)
    
    dic_loss[str(lr)] = history.history['loss']
    lr_.append(lr)
    mae_.append(mae)
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

df = pd.DataFrame(dic_loss)
df.plot().grid()
plt.semilogx(lr_, mae_)