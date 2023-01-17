# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/renfe.csv", index_col =0, parse_dates = ['insert_date', 'start_date', 'end_date'], infer_datetime_format = True)
df

df.dtypes
df.isna().sum()
df = df.dropna(how = 'any',subset = ['price'])
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize = (10,8))

sns.barplot(x='origin', y = 'price', data = df)

plt.xlabel('Origin')

plt.ylabel('Avg. Price')
plt.figure(figsize = (10,8))

sns.countplot(x='origin', data = df)

plt.xlabel('Origin')

plt.title('Station wise purchase of tickets')
mad = df.loc[(df['origin']=='MADRID'),:]



sev = df.loc[(df['origin']=='SEVILLA'),:]



pon = df.loc[df['origin']=='PONFERRADA',:]



barca = df.loc[(df['origin']=='BARCELONA'),:]



val = df.loc[(df['origin']=='VALENCIA'),:]
sns.set(style="whitegrid")

plt.figure(figsize = (20,30))

g = sns.catplot(x = 'train_type', y='price',hue = 'train_class', data=mad, col='destination', col_wrap = 2, kind = 'bar')

g.set_xticklabels(rotation=90)
sns.set(style="whitegrid")

plt.figure(figsize = (20,30))

g = sns.catplot(x = 'train_type', y='price', hue = 'train_class', data=sev, col='destination', col_wrap = 2, kind = 'bar')

g.set_xticklabels(rotation=90)
sns.set(style="whitegrid")

plt.figure(figsize = (20,30))

g = sns.catplot(x = 'train_type', y='price', hue='train_class', data=pon, col='destination', col_wrap = 2, kind = 'bar')

g.set_xticklabels(rotation=90)
sns.set(style="whitegrid")

plt.figure(figsize = (20,30))

g = sns.catplot(x = 'train_type', y='price', hue='train_class', data=barca, col='destination', col_wrap = 2, kind = 'bar')

g.set_xticklabels(rotation=90)
sns.set(style="whitegrid")

plt.figure(figsize = (20,30))

g = sns.catplot(x = 'train_type', y='price', hue='train_class', data=val, col='destination', col_wrap = 2, kind = 'bar')

g.set_xticklabels(rotation=90)
plt.figure(figsize = (16,8))

sns.barplot(x='train_class', y = 'price', data = df)

plt.xlabel('Train Class')

plt.ylabel('Avg. Price')
plt.figure(figsize = (16,8))

g = sns.countplot(x = 'train_class', data = df)

plt.xticks(rotation=90)

plt.xlabel('Train Class')

plt.title('Popularity of Train Classes')
plt.figure(figsize = (16,8))

g = sns.countplot(x = 'train_type', data = df)

plt.xticks(rotation=90)

plt.xlabel('Train Type')

plt.title('Popularity of Train Types')
plt.figure(figsize = (16,8))

sns.barplot(x='train_type', y = 'price', data = df)

plt.xticks(rotation = 90)

plt.xlabel('Train Type')

plt.ylabel('Avg. Price')

plt.title('Price distribution of Train types')
plt.figure(figsize = (16,8))

sns.barplot(x='fare', y = 'price', data = df)

plt.xticks(rotation = 90)

plt.xlabel('Fare Type')

plt.ylabel('Avg. Price')

plt.title('Price Distribution of Fare Types')
plt.figure(figsize = (16,8))

sns.countplot(x='fare', data = df)

plt.xticks(rotation = 90)

plt.xlabel('Fare Type')

plt.title('Popularity of Fare Types')
plt.figure(figsize = (10,8))

sns.distplot(df['price'])

plt.xlabel('Price')

plt.title('Distribution of Price')
def eta(z):

    start = z['start_date']

    end = z['end_date']

    td = end - start

    days = td.days

    hours, remainder = divmod(td.seconds, 3600)

    minutes, seconds = divmod(remainder, 60)

    total_hours = hours + minutes/60

    return total_hours

df['travelTime'] = df.apply(eta, axis = 1)
plt.figure(figsize=(10,8))

sns.distplot(df['travelTime'])

plt.xlabel('Travel Time')

plt.title('Distribution of Travel Times')
def booking(z):

    start = z['start_date']

    book = z['insert_date']

    td = start - book

    days = td.days

    hours, remainder = divmod(td.seconds, 3600)

    minutes, seconds = divmod(remainder, 60)

    total_days = days + hours/24 + minutes/60

    return total_days

df['bookingProximity'] = df.apply(booking, axis = 1)
df = df.loc[df['bookingProximity']>0,:]
plt.figure(figsize=(10,8))

sns.lineplot(x='bookingProximity', y ='price', data = df)
plt.figure(figsize=(10,8))

sns.distplot(df['bookingProximity'])
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from sklearn.model_selection import train_test_split
miT = min(x['travelTime'])

maT = max(x['travelTime'])

x['normTravelTime']=[(i-miT)/(maT-miT) for i in x['travelTime']]
miB = min(x['bookingProximity'])

maB = max(x['bookingProximity'])

x['normbookingProximity']=[(i-miB)/(maB-miB) for i in x['bookingProximity']]
X = x.drop(['bookingProximity', 'travelTime'], axis = 1)
X = pd.get_dummies(X)
mi = min(y)

ma = max(y)

y = [(i-mi)/(ma-mi) for i in y]
x = df[['train_type', 'train_class', 'fare', 'origin', 'destination', 'travelTime', 'bookingProximity']]

y = df['price']
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.30, random_state=4)
rf = RandomForestRegressor(n_estimators = 100)

rf.fit(x_train, y_train)
pred = rf.predict(x_test)
from sklearn import metrics

mse = metrics.mean_squared_error(y_test, pred)

print(mse)

r2 = metrics.r2_score(y_test, pred)

print(r2)