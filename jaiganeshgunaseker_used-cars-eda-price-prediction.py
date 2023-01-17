# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')

df.head()
df.info()
df = df[['region', 'price', 'year', 'manufacturer','model', 'condition', 'cylinders', 'fuel', 'odometer', 'title_status','transmission', 'drive', 'size', 'type', 'paint_color','state']]
print(df.isnull().sum().sort_values(ascending = False))

df.isnull().sum().sort_values().plot(kind='barh')

plt.axvline(x=df.shape[0],color='red',linestyle='--')

plt.axvline(x=df.shape[0]/2,color='orange',linestyle='--')

plt.axvline(x=df.shape[0]/4,color='yellow',linestyle='--')

plt.text(df.shape[0]/4,6,'25%',rotation=90)

plt.text(df.shape[0]/2,6,'50%',rotation=90)

plt.text(df.shape[0],6,'Total observations',rotation=90)
df = df.dropna()
plt.figure(figsize=(25,5))

df['year'].value_counts().sort_values().head(100).plot(kind = 'bar',color='orange')
plt.figure(figsize=(10,20))

df['region'].value_counts().sort_values().head(100).plot(kind = 'barh',color='teal')
plt.figure(figsize=(15,6))

df['manufacturer'].value_counts().sort_values().plot(kind = 'bar',color='red')
plt.figure(figsize=(5,3))

df['condition'].value_counts().sort_values().plot(kind='barh',color='green')
condition_price = pd.pivot_table(data=df,values='price',index='condition',aggfunc='mean').sort_values('price',ascending=False).reset_index()

sns.barplot(y='condition' , x='price',data=condition_price)
plt.figure(figsize=(10,5))

sns.countplot(df.cylinders)
plt.figure(figsize=(10,4))

max_price = pd.pivot_table(data=df,values='price',index='cylinders',aggfunc='mean').sort_values('price',ascending= False).reset_index()

sns.barplot(y='cylinders',x='price',data=max_price,color='gold')
plt.figure(figsize=(10,4))

max_price = pd.pivot_table(data=df,values='price',index='paint_color',aggfunc='mean').sort_values('price',ascending= False).reset_index()

sns.barplot(y='paint_color',x='price',data=max_price,color='brown')
plt.figure(figsize=(6,3))

print(df['fuel'].value_counts().sort_values(ascending =False))

df['fuel'].value_counts().sort_values().plot(kind = 'barh',color='black')
df[df['cylinders'] == 'other']
df = df[df['cylinders'] != 'other']
df[df['fuel'] == 'other']
df = df[df['fuel'] != 'other']
plt.figure(figsize=(15,10))

max_price = pd.pivot_table(data=df,values='price',index='manufacturer',aggfunc='max').sort_values('price',ascending= False).reset_index()

sns.barplot(y='manufacturer',x='price',data=max_price)
plt.figure(figsize=(20,10))

sns.boxplot(x='price',y='manufacturer',data=df)

plt.axvline(x=0.001e+9,linestyle='--')
plt.figure(figsize=(20,10))

sns.boxplot(x='price',y='manufacturer',data=df[df['price']<=0.001e+9])
df = df[df['price']<=0.001e+9]
plt.figure(figsize=(20,10))

sns.boxplot(x='price',y='manufacturer',data=df)

plt.axvline(x=20200,linestyle='--',label = 'Current Mean Price from Google')

plt.axvline(x=df['price'].mean(),linestyle='--',color='red',label='Mean price in the dataset')

plt.axvline(x=df['price'].median(),linestyle='--',color='yellow',label = 'Median price in the dataset')

plt.legend()
def outliers(col):

    q1 = np.quantile(col,0.25)

    q3 = np.quantile(col,0.75)

    iqr = q3 - q1

    lrlimit = q1 - (1.5 * iqr)

    urlimit = q3 + (1.5 * iqr)

    return lrlimit , urlimit
outliers(df['price'])
print('Number of cars where price is an outlier :',df[df['price'] > 31462.5].shape[0])
print('Number of cars where price is 0 :',df[df['price'] == 0].shape[0])
df = df[df['price']!=0]
df = df[df['price'] <= 31462.5]
df.shape
for i in df.columns:

    if df[i].dtype == 'object':

        print(i,' ',(df[i].value_counts()).shape[0],'\n',df[i].value_counts(),'\n\n')
df = pd.get_dummies(df.drop(['model'],axis=1),drop_first=False)
from sklearn.model_selection import train_test_split
X = df.drop('price',axis=1)



Y = df['price']
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.3 , random_state = 1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor

import xgboost as xgb

from sklearn.ensemble import AdaBoostRegressor
lrmodel = MLPRegressor()

lrmodel.fit(X_train_scaled , Y_train)

print('Training data score',lrmodel.score(X_train_scaled , Y_train),'\n')

print('Test data score',lrmodel.score(X_test_scaled , Y_test),'\n')
lrmodel = xgb.XGBRegressor()

lrmodel.fit(X_train , Y_train)

print(lrmodel.score(X_train , Y_train),'\n')

print(lrmodel.score(X_test , Y_test),'\n')
abmodel = AdaBoostRegressor()

abmodel.fit(X_train , Y_train)

print(abmodel.score(X_train , Y_train),'\n')

print(abmodel.score(X_test , Y_test),'\n')