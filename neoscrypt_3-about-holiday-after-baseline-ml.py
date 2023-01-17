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

df=pd.read_csv('/kaggle/input/eda-df/eda_df.csv')
df.head()
holiday=pd.read_csv('/kaggle/input/for-holiday-1/2011_2012_real_holiday.csv')
holiday.head()
df['ymd'] = df['year_month'] + '-' + df['day'].astype(str)
df.head()
interested = ['holiday_name', 'holiday', 'ymd']
for_join = holiday[interested]
for_join.head()
df = pd.merge(df, for_join, on='ymd', how='left')
df.head()
df.describe()
df['holiday'] = df['holiday'].fillna(0)
df['holiday_name'] = df['holiday_name'].fillna('no_holiday')
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['weekend'] == 0]
sns.pointplot(data=target, x="hour", y="count", hue="holiday", ax=ax1)

figure, ax2 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['weekend'] == 1]
sns.pointplot(data=target, x="hour", y="count", hue="holiday", ax=ax2)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['weekend'] == 0]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday", ax=ax1)

figure, ax2 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['weekend'] == 1]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday", ax=ax2)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['weekend'] == 0]
sns.pointplot(data=target, x="hour", y="casual", hue="holiday", ax=ax1)

figure, ax2 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['weekend'] == 1]
sns.pointplot(data=target, x="hour", y="casual", hue="holiday", ax=ax2)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

sns.pointplot(data=df, x="dayofweek", y="count", hue="holiday", ax=ax1)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

sns.pointplot(data=df, x="dayofweek", y="registered", hue="holiday", ax=ax1)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

sns.pointplot(data=df, x="dayofweek", y="casual", hue="holiday", ax=ax1)
# checkout wednesday(dayofweek = 2)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['dayofweek'] == 2]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday_name", ax=ax1)
# checkout mon, tuesday
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['dayofweek'] == 0]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday_name", ax=ax1)

figure, ax2 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['dayofweek'] == 1]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday_name", ax=ax2)
# checkout thurs, friday
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['dayofweek'] == 3]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday_name", ax=ax1)

figure, ax2 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['dayofweek'] == 4]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday_name", ax=ax2)
# checkout weekends (sat, sunday)
figure, ax1 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['dayofweek'] == 5]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday_name", ax=ax1)

figure, ax2 = plt.subplots(nrows=1, ncols=1)
figure.set_size_inches(18, 4)

target = df.loc[df['dayofweek'] == 6]
sns.pointplot(data=target, x="hour", y="registered", hue="holiday_name", ax=ax2)
# holidays aren't the same holidays!
by_holiday = df.groupby('holiday_name')
popular_h = by_holiday.agg({'registered' : np.mean}).reset_index()
popular_h.head()
popular_h.sort_values(by='registered', ascending=False)
popular_h.columns = ['holiday_name', 'holiday_index']
popular_h.head()
df = pd.merge(df, popular_h, how='left', on='holiday_name')
df.head()
df.describe()
# deploy ML with holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(n_estimators=20, n_jobs=-1)
model
def df_split(df, interested):
    label = 'count'
    train, test = df[:7000], df[7000:]
    X_train, X_test = train[interested], test[interested]
    y_train, y_test = train[label], test[label]
    return X_train, X_test, y_train, y_test
# we were here from 2. baseline_ml
interested = ['year', 'month', 'day', 'hour', 'workingday', 'dayofweek', 'weekend']
X_train, X_test, y_train, y_test = df_split(df, interested)

score = cross_val_score(model, X_train, y_train, cv=20).mean()

print("Score = {0:.5f}".format(score))
# add holiday
interested = ['year', 'month', 'day', 'hour', 'workingday', 'dayofweek', 'weekend', 'holiday']
X_train, X_test, y_train, y_test = df_split(df, interested)

score = cross_val_score(model, X_train, y_train, cv=20).mean()

print("Score = {0:.5f}".format(score))
# add holiday_index
interested = ['year', 'month', 'day', 'hour', 'workingday', 'dayofweek', 'weekend', 'holiday', 'holiday_index']
X_train, X_test, y_train, y_test = df_split(df, interested)

score = cross_val_score(model, X_train, y_train, cv=20).mean()

print("Score = {0:.5f}".format(score))
# checkout feature importances
model.fit(X_train, y_train)
sorted(zip(model.feature_importances_, X_train.columns))
# again, hour seems the best describing feature among ALL.



