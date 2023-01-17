import pandas as pd

import numpy as np

import seaborn as sns

import os

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('/kaggle/input/happiness.csv')

df.head()
df.isnull().count()
df.isnull().sum()
df=df.rename(columns={'Country(region)':'Country','SD of Ladder':'SD','Positive affect':'Positive','Negative affect':'Negative',

                      'Log of GDP\nper capita':'GDP'})
df.head()
df.info()
df=df.fillna(method='ffill')
df.describe()
df.GDP.head()
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(),ax=ax,annot=True,linewidth=0.05,fmt='.2f',cmap='magma')

plt.show()

y = df['Social support']

x =df['Ladder']

plt.scatter(x,y)

plt.xlabel('Ladder')

plt.ylabel('Social Support')

plt.title('LADDER AND SOCIAL SUPPORT')
y = df['GDP']

x =df['Healthy life\nexpectancy']

plt.scatter(x,y)

plt.xlabel('GDP')

plt.ylabel('Healthy life')

plt.title('GDP AND HEALTH LIFE')
from sklearn.model_selection import  train_test_split

from sklearn.linear_model import  LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score 
df.head()
x = df.drop(['SD','Country (region)'],axis=1)

y = df['SD']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

df.head()
lr=LinearRegression()

lr.fit(x_train,y_train)

predict_lr=lr.predict(x_test)

print('r2 score:' , r2_score(y_test,predict_lr))

plt.bar(y_test,predict_lr,color='c')

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('LinearRegression')
d=DecisionTreeRegressor()

d.fit(x_train,y_train)

predict_d=d.predict(x_test)

print('r2_score:',r2_score(y_test,predict_d))

plt.bar(y_test,predict_d)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('DecisionTreeRegressor')
knn=KNeighborsRegressor()

knn.fit(x_train,y_train)

predict_knn=knn.predict(x_test)

print('r2_score:',r2_score(y_test,predict_knn))

plt.bar(y_test,predict_knn)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('KNeighborsRegressor')
rnd=RandomForestRegressor()

rnd.fit(x_train,y_train)

predict_rnd=rnd.predict(x_test)

print('r2_score',r2_score(y_test,predict_rnd))

plt.bar(y_test,predict_rnd)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('RandomForestRegressor')
y = np.array([r2_score(y_test,predict_lr),r2_score(y_test,predict_rnd),r2_score(y_test,predict_d),

              r2_score(y_test,predict_knn)])

              

              

x = ['Linear','RandomForest','DecisionTree','KNNR']



plt.bar(x,y)

plt.title('comparision')

plt.xlabel('Regressor')

plt.ylabel('r2_score')