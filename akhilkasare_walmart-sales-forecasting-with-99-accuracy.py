# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm 

import pylab as py 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')

test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')

stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')

feature = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
train.head()
test.head()
train.info()
train.describe()
# Checking for null values

train.isnull().sum()
test.isnull().sum()
train['year'] = pd.DatetimeIndex(train['Date']).year

train['month'] = pd.DatetimeIndex(train['Date']).month

train['day'] = pd.DatetimeIndex(train['Date']).day
train.drop(['Date'], axis=1,inplace=True)
train.head()
sns.distplot(train['Weekly_Sales'])
train.Weekly_Sales=np.where(train.Weekly_Sales>100000, 100000,train.Weekly_Sales)
train.Weekly_Sales.plot.hist(bins=25)
train.IsHoliday.value_counts()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



train['IsHoliday'] = le.fit_transform(train['IsHoliday'])
train.head()
train.hist(edgecolor='black', linewidth=1.2, figsize=(20, 20));
train.head()
plt.figure(figsize=(10,7))

sns.barplot(x='day', y='Weekly_Sales', data=train)
plt.figure(figsize=(10,7))

sns.barplot(x='month', y='Weekly_Sales', data=train)
plt.figure(figsize=(10,7))

sns.barplot(x='year', y='Weekly_Sales', data=train)
# 1 : Sales during Holidays

# 0 : Sales during Non-Holidays



plt.figure(figsize=(10,7))

train.groupby('IsHoliday')['Weekly_Sales'].mean().plot(kind = 'barh')

plt.title('Sales During Holidays and Non-Holidays')

plt.xlabel('Sales')
weekly_sales_mean = train['Weekly_Sales'].groupby(train['day']).mean()

weekly_sales_median = train['Weekly_Sales'].groupby(train['day']).median()

plt.figure(figsize=(20,8))

sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values)

sns.lineplot(weekly_sales_median.index, weekly_sales_median.values)

plt.grid()

plt.legend(['Mean', 'Median'], loc='best', fontsize=16)

plt.title('Weekly Sales - Mean and Median', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Date', fontsize=16)

plt.show()
weekly_sales = train['Weekly_Sales'].groupby(train['Store']).mean()

plt.figure(figsize=(20,8))

sns.barplot(weekly_sales.index, weekly_sales.values, palette='dark')

plt.grid()

plt.title('Average Sales - per Store', fontsize=18)

plt.ylabel('Sales', fontsize=16)

plt.xlabel('Store', fontsize=16)

plt.show()
plt.figure(figsize=(10, 10))

sns.heatmap(train.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
# Formatting the Date cloumn



test['year'] = pd.DatetimeIndex(test['Date']).year

test['month'] = pd.DatetimeIndex(test['Date']).month

test['day'] = pd.DatetimeIndex(test['Date']).day



# Dropping the Date column

test.drop(['Date'], axis=1,inplace=True)



# Label Encoding the IsHoliday column



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



test['IsHoliday'] = le.fit_transform(test['IsHoliday'])
test.head()
x = train.drop(['Weekly_Sales'],axis=1)

y = train['Weekly_Sales']

x_test = test
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestRegressor

clf_rf = RandomForestRegressor(n_estimators=150)

clf_rf.fit(x_train, y_train)

y_pred_rf=clf_rf.predict(x_val)

acc_rf= round(clf_rf.score(x_train, y_train) * 100, 2)

print ("Accuracy: %i %% \n"%acc_rf)
rmse = np.sqrt(mean_squared_error(y_pred_rf, y_val))

rmse
y_pred = clf_rf.predict(x_test)

y_pred
pred = pd.DataFrame(y_pred)



sub = pd.read_csv('../input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv.zip')



sub['Weekly_Sales'] = pred

sub.to_csv('submission.csv', index=False)
a=pd.read_csv('submission.csv')

a.head()