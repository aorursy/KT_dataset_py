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
data = pd.read_csv("/kaggle/input/beer-consumption-sao-paulo/Consumo_cerveja.csv", decimal=',')

data.head()
data.columns = ['date', 'TempMedian', 'TempMin','TempMax', 'Precipitation','isWeekend','consumption']

data.info()
data.shape




data = data.dropna()

data.shape
data.info()
data['consumption'] = data['consumption'].astype(float)

data.info()
data.plot(kind='scatter', x='TempMedian', y= 'consumption' )
data.plot(kind='scatter', x='TempMin', y= 'consumption' )
data.plot(kind='scatter', x='TempMax', y= 'consumption' )
data.plot(kind='scatter', x='Precipitation', y= 'consumption' )
data.plot(kind='scatter', x='isWeekend', y= 'consumption' )
from sklearn.linear_model import LinearRegression 

from sklearn.model_selection import train_test_split
feature_cols = ['TempMedian',  'Precipitation','isWeekend']

X = data[feature_cols]

Y = data.consumption

X_train, X_test, Y_train, Y_Test = train_test_split(X,Y, random_state =1)
model = LinearRegression()

model.fit(X_train,Y_train)


# print coefficients

print(feature_cols, model.coef_)
model.score(X_test,Y_Test)