%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/world-happiness-report-2019.csv')
df = df.rename(columns = {'Country (region)':'Country','SD of Ladder':'SD',

                         'Positive affect':'Positive','Negative affect':'Negative','Social support':'Social',

                         })
df.head()
df.info()
df.isnull().sum()
df = df.fillna(method = 'ffill')
df.describe()
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(),ax=ax,annot=True,linewidth=0.05,fmt='.2f',cmap='magma')

plt.show()
plt.scatter(df['Positive'],df.Negative)

plt.title('Positive compare with Negative')

plt.xlabel('Positive')

plt.ylabel('Negative')

plt.show()
y = np.array([df['Positive'].min(),df['Positive'].mean(),df['Positive'].max()])

x = ['lowest','average','highest']

plt.bar(x,y)

plt.xlabel('Lavel')

plt.ylabel('Positive')

plt.title('Positive')
plt.scatter(df['Positive'],df.Social)

plt.xlabel('Positive')

plt.ylabel('Social')

plt.title('Positive compare with Negative')
plt.scatter(df['Positive'],df.Freedom)

plt.xlabel('Positive')

plt.ylabel('Freedom')

plt.title('Positive compare with Freedom')
plt.scatter(df['Positive'],df.Generosity)

plt.xlabel('Positive')

plt.ylabel('Generosity')

plt.title('Positive compare with Generosity')
plt.scatter(df['Negative'],df.Corruption)

plt.xlabel('Negative')

plt.ylabel('Corruption')

plt.title('Negative compare with Corruption')
df.sort_values(by = ['SD'],ascending=False).plot(x='Country',y='SD')

plt.xlabel('Ladder')

plt.ylabel('SD')
sns.jointplot('Positive','Negative',data=df,kind='kde')
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score
x = df.drop(['SD','Country'],axis=1)

y = df['SD']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
lr = LinearRegression()

lr.fit(x_train,y_train)

predict_lr = lr.predict(x_test)

print('r2 score:' , r2_score(y_test,predict_lr))

plt.scatter(y_test,predict_lr,color='c')

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('LinearRegression')
rfg = RandomForestRegressor()

rfg.fit(x_train,y_train)

predict_rfg = rfg.predict(x_test)

print('r2 score:' , r2_score(y_test,predict_rfg))

plt.scatter(y_test,predict_rfg)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('RandomForestRegressor')
dtr = DecisionTreeRegressor()

dtr.fit(x_train,y_train)

predict_dtr = dtr.predict(x_test)

print('r2 score:' , r2_score(y_test,predict_dtr))

plt.scatter(y_test,predict_dtr)

plt.xlabel('y in test')

plt.ylabel('prediction')

plt.title('DecisionTreeRegressor')
y = np.array([r2_score(y_test,predict_lr),r2_score(y_test,predict_rfg),r2_score(y_test,predict_dtr)])

x = ['Linear','RandomForest','DecisionTree']



plt.bar(x,y)

plt.title('comparision')

plt.xlabel('Regressor')

plt.ylabel('r2_score')