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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
df = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/vehicles.csv')
df.head(10)
print(df.columns)
print(df.shape)
df.isna().sum()
def remove_col(data):
    thresh = len(data) * 0.4
    cols = data.columns
    remove = []
    for col in cols:
        n_nulls = data[col].isna().sum()
        if n_nulls >= thresh:
            remove.append(col)
    return remove

rm_cols = remove_col(df)
df = df.drop(rm_cols,axis=1)
df.head(5)
df.nunique()
rm_cols = [
    'id',
    'url',
    'region',
    'region_url',
    'image_url',
    'description',
    'model',
    'state',
    'paint_color'
]
df = df.drop(rm_cols,axis=1)
df.head(10)
print(df.price.describe())
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.price)
descp = interquartile = df.price.describe()
interquartile = descp['75%'] - descp['25%']
thresh = interquartile * 1.5

df = df[df.price < thresh]
df.head(3)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.price)
df[['odometer','year']].describe()
df.year.value_counts()
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.countplot(df[df.year.between(1950,2020)].year)
# plt.xticks([0,15,30,45,60,70])
df = df[df.year.between(1960,2020)]
df.head(3)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.odometer)
interquartile = df.odometer.quantile(0.75) - df.odometer.quantile(0.25)
thresh = interquartile * 1.5
df = df[df.odometer < thresh]
df.head(3)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.boxplot(df.odometer)
top_manufacturers = df.manufacturer.value_counts(dropna=False).iloc[:10]
print(top_manufacturers)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.barplot(x=top_manufacturers.index,y=top_manufacturers.values)
plt.xlabel('Manufacturers')
plt.ylabel('Number of vehicles')
plt.title('Vehicles from top 10 manufacturers',y=1.02)
plt.suptitle('Number of vehicles from the top 10 manufacturers listed on Craigslist',y=0.9)
top_types = df.type.value_counts().iloc[:10]
print(top_types)

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
sns.barplot(x=top_types.index,y=top_types.values)
plt.xlabel('Vehicle type')
plt.ylabel('Number of vehicles')
plt.title('Generic top 10 vehicle types',y=1.02)
plt.suptitle('Number of vehicles from the top 10 types of generic vehicle models listed on Craigslist',y=0.9)
df = df.dropna(subset=['lat','long'])
df.head(5)
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, annot=True,cmap='YlGnBu')
df_cleaned = pd.get_dummies(df)
X = df_cleaned.iloc[:,1:]
y = df_cleaned.price
X.columns
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
model = RandomForestRegressor(n_estimators=25,random_state=0)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
pred = model.predict(X_test)
print(mae(y_test,pred))
print(y.mean())
model.score(X_test,y_test)
feature_imp = pd.Series(model.feature_importances_,index=df_cleaned.columns[1:])
plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
feature_imp.sort_values(ascending=False)[:20].plot.barh()
plt.ylabel('Features')
plt.xticks([])