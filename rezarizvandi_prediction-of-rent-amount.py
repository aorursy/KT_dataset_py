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
data = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent.csv')
data.head()
data.drop('Unnamed: 0',axis = 1 , inplace = True)
data['floor'] = data['floor'].replace('-', 0)
data ['animal'] = data['animal'].replace('acept', 1)

data ['animal'] = data['animal'].replace('not acept', 0)

data['furniture'] = data['furniture'].replace('furnished' , 1)

data['furniture'] = data['furniture'].replace('not furnished' , 0)
data.drop('hoa',axis = 1 , inplace =True)

data.drop('total',axis = 1 , inplace = True)

data.drop('property tax',axis = 1 , inplace = True)

data.drop('fire insurance',axis = 1 , inplace = True)
import re

data['rent amount'] = data['rent amount'].map(lambda x: re.sub(r'\D+', '', x))

x = data.drop('rent amount' , axis = 1)

y = data['rent amount']

x = x.values

y = y.values

x = x.astype(float)

y = y.astype(float)
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import ExtraTreeRegressor 

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error,mean_squared_error
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = 0.2)
regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
print("regressor score in Training", regressor.score(x_train , y_train))

rscore = r2_score(y_test,y_pred)

print("r2score is", rscore)

mae = mean_absolute_error(y_test,y_pred)

print("mean absolute error is",mae)

mse = mean_squared_error(y_test,y_pred)

print("root mean squared error is",np.sqrt(mse))
regressor = GradientBoostingRegressor()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#print("regressor name",regressor.__str__)

print("regressor score in Training", regressor.score(x_train , y_train))

rscore = r2_score(y_test,y_pred)

print("r2score is", rscore)

mae = mean_absolute_error(y_test,y_pred)

print("mean absolute error is",mae)

mse = mean_squared_error(y_test,y_pred)

print("root mean squared error is",np.sqrt(mse))

regressor = RandomForestRegressor()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#print("regressor name",regressor.__str__)

print("regressor score in Training", regressor.score(x_train , y_train))

rscore = r2_score(y_test,y_pred)

print("r2score is", rscore)

mae = mean_absolute_error(y_test,y_pred)

print("mean absolute error is",mae)

mse = mean_squared_error(y_test,y_pred)

print("root mean squared error is",np.sqrt(mse))
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

model = Sequential()

model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))

model.add(Dense(15  ,kernel_initializer = 'normal' , activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(15 , kernel_initializer = 'normal' , activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(15 , kernel_initializer = 'normal' , activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, kernel_initializer='normal' , activation='linear'))

# Compile model

model.compile(loss='mean_squared_error', optimizer='adam')



model.fit(x_train , y_train , epochs = 150 , batch_size=5)

y_pred = model.predict(x_test)



mae = mean_absolute_error(y_test,y_pred)

print("mean absolute error is",mae)

mse = mean_squared_error(y_test,y_pred)

print("root mean squared error is",np.sqrt(mse))