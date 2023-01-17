import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/avocado-prices/avocado.csv')

df
df.drop(['XLarge Bags'],axis=1,inplace=True)

df.drop(['Unnamed: 0','Date'],axis=1,inplace=True)
plt.hist(df['AveragePrice'],facecolor='violet',edgecolor='black',bins=10)
plt.figure(figsize=(12,6))

sns.boxplot(x=df['region'], y=df['AveragePrice']);

plt.xticks(rotation=80)
plt.hist(df['year'])
df.isnull().sum()
df.type.unique()

df.region.unique()
df1=pd.get_dummies(df['type'],drop_first=True)

df2=pd.get_dummies(df['region'],drop_first=True)
df5=pd.concat([df,df1,df2],axis=1)

df5.head()
df5.drop(['type','region'],axis=1,inplace=True)
import sklearn

from sklearn.model_selection import train_test_split
X=df5.loc[:, 'Total Volume':'WestTexNewMexico']

y=df5[['AveragePrice']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
import xgboost as xgb

from sklearn.metrics import r2_score

xgb = xgb.XGBRegressor()
xgb.fit(X_train,y_train)
model=xgb.predict(X_test)

r2_score(y_test,model)
from sklearn.ensemble import RandomForestRegressor

RR=RandomForestRegressor(random_state=1)

RR.fit(X_train,y_train)
model1=RR.predict(X_test)

r2_score(y_test,model1)