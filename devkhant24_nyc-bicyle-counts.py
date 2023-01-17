# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.feature_selection import SelectKBest

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df =pd.read_csv('/kaggle/input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['date'] = df['Date'].dt.day

df.drop('Date',axis=1,inplace=True)
df['Day'] = pd.to_datetime(df['Day'])

df['day'] = df['Day'].dt.day_name()

df.drop('Day',axis=1,inplace=True)
#df['Day'].equals(df['Date'])

#with pd.option_context('display.max_rows', None, 'display.max_columns', None): 

    #print(df['Date'])
dummies = pd.get_dummies(df['day'],drop_first=True) 
df = pd.concat([dummies,df],axis=1)
df.drop('day',axis=1,inplace=True)
x1 = df[['Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday','High Temp (°F)','Low Temp (°F)','date']]

y1 = df['Brooklyn Bridge']

x2 = df[['Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday','High Temp (°F)','Low Temp (°F)','date']]

y2 = df['Manhattan Bridge']

x3 = df[['Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday','High Temp (°F)','Low Temp (°F)','date']]

y3 = df['Queensboro Bridge']

x4 = df[['Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday','High Temp (°F)','Low Temp (°F)','date']]

y4 = df['Williamsburg Bridge']
#xg = xgboost.XGBRegressor(objective='reg:squarederror')

#xg.fit(x1,y1)
lr = LinearRegression()

pr = PolynomialFeatures(degree=2)

xpr1 = pr.fit_transform(x1)

xpr2 = pr.fit_transform(x2)

xpr3 = pr.fit_transform(x3)

xpr4 = pr.fit_transform(x4)
lr.fit(xpr1,y1)

yp1 = lr.predict(xpr1)

pred1 = pd.DataFrame(yp1)
lr.fit(xpr2,y2)

yp2 = lr.predict(xpr2)

pred2= pd.DataFrame(yp2)  
lr.fit(xpr3,y3)

yp3 = lr.predict(xpr3)

pred3= pd.DataFrame(yp3)  
lr.fit(xpr4,y4)

yp4 = lr.predict(xpr4)

pred4= pd.DataFrame(yp4)  
df['sum'] = pred1+pred2+pred3+pred4
df.drop('Total',axis=1,inplace=True)