import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt

import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns 

% matplotlib inline
from sklearn.linear_model import LinearRegression

import statsmodels.formula.api as sm

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
df=pd.read_csv('../input/GDP_CAT.csv')
df = df.iloc[::-1]
df = df.set_index('Year')
df.head()
X=['Consumer expenditure household','Consumer public adm','Equip. Goods others','Const.',

   'Total exports goods and services','Total imports goods and services']

X=df[X]

y=df.GDP

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.score(X_test,y_test))

print(sqrt(mean_squared_error(y_test,lm.predict(X_test))))
Cons_per_GDP=df['Const.']/df.GDP*100

Exports_per_GDP=df['Total exports goods and services']/df.GDP*100

df['Cons_per_GDP']=Cons_per_GDP

df['Exports_per_GDP']=Exports_per_GDP

Domestic_Demand_per_GDP_wc=(df['Domestic demand']-df['Const.'])/df.GDP*100

df['Domestic_Demand_per_GDP_wc']=Domestic_Demand_per_GDP_wc

df['trad_op']= (df['Total exports goods and services']+df['Total imports goods and services'])

df['trad_op']= (df['trad_op']/df.GDP*100) 

df['pct_change']=(df.GDP.pct_change()*100)        
X=['Consumer expenditure household','Consumer public adm','Equip. Goods others','Const.',

   'Total exports goods and services','Total imports goods and services','trad_op',

   'Domestic_Demand_per_GDP_wc'] 

X=df[X]

y=df.GDP

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

lm = LinearRegression()

lm.fit(X_train,y_train)

print(lm.score(X_test,y_test))

print(sqrt(mean_squared_error(y_test,lm.predict(X_test))))