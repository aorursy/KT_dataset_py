# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')

df.head()
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
plt.figure(figsize=(12,8))

sns.boxplot(df.price)
descp = interquartile = df.price.describe()

interquartile = descp['75%'] - descp['25%']

thresh = interquartile * 1.5



df = df[df.price < thresh]

df.head(3)
plt.figure(figsize=(12,8))

sns.boxplot(df.price)
df = df[df.year.between(1960,2020)]

df.head(3)


plt.figure(figsize=(12,8))

sns.boxplot(df.odometer)
interquartile = df.odometer.quantile(0.75) - df.odometer.quantile(0.25)

thresh = interquartile * 1.5

df = df[df.odometer < thresh]

df.head(3)
plt.figure(figsize=(12,8))

sns.boxplot(df.odometer)
top_manufacturers = df.manufacturer.value_counts(dropna=False).iloc[:10]

print(top_manufacturers)



plt.figure(figsize=(12,8))

sns.barplot(x=top_manufacturers.index,y=top_manufacturers.values)

plt.xlabel('Производители')

plt.ylabel('Кол-во автомобилей')

plt.title('Топ-10 производителей',y=1.02)
df = df.dropna(subset=['lat','long'])

df.head(5)
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

print(y.mean())

model.score(X_test,y_test)