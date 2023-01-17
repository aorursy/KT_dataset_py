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
import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std

import math

import sklearn.preprocessing as skpe

import sklearn.model_selection as ms

import sklearn.metrics as sklm

import sklearn.ensemble as sken

import sklearn.linear_model as lm

import seaborn as sns

import matplotlib.pyplot as plt
# Reading files

path="../input/rossmann-store-sales/train.csv"

train=pd.read_csv(path)

print(train.shape)

train.head()
path1="../input/rossmann-store-sales/test.csv"

test=pd.read_csv(path1)

print(test.shape)

test.head()
path2="../input/rossmann-store-sales/store.csv"

store_df=pd.read_csv(path2)

print(store_df.shape)

store_df.head()
train.info()

print("----------------------------------------------")



store_df.info()

print("----------------------------------------------")



test.info()
train.describe()
# Adding new variable

train['Sales_per_customer']=train['Sales']/train['Customers']

train['Sales_per_customer'].describe() # An average of 9.49$ is earned from a customer at a particular store
fig, ax1 = plt.subplots(figsize=(15,4))

sns.countplot(x='Open',hue='DayOfWeek', data=train,palette="husl", ax=ax1) # This indicates that there are some stores which opens mostly on Sundays while some are closed on Sundays 
# Date



# Create Year and Month columns

train['Year']  = train['Date'].apply(lambda x: int(str(x)[:4]))

train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))



test['Year']  = test['Date'].apply(lambda x: int(str(x)[:4]))

test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))



# Assign Date column to Date(Year-Month) instead of (Year-Month-Day)

train['Date'] = train['Date'].apply(lambda x: (str(x)[:7]))

test['Date']     = test['Date'].apply(lambda x: (str(x)[:7]))



# group by date and get average sales, and percent change

avg_sales    = train.groupby('Date')["Sales"].mean()

pct_change_sales = train.groupby('Date')["Sales"].sum().pct_change()



fig, (axis1,axis2) = plt.subplots(2,1,sharex=True,figsize=(15,8))



# plot average sales over time(year-month)

ax1 = avg_sales.plot(legend=True,ax=axis1,marker='o',title="Average Sales")

ax1.set_xticks(range(len(avg_sales)))

ax1.set_xticklabels(avg_sales.index.tolist(), rotation=90)



# plot precent change for sales over time(year-month)

ax2 = pct_change_sales.plot(legend=True,ax=axis2,marker='o',rot=90,colormap="summer",title="Sales Percent Change")
# Plot average sales and customers over years

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Year', y='Sales', data=train, ax=axis1)

sns.barplot(x='Year', y='Customers', data=train, ax=axis2)
# Plot average sales and customers over days of week

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='DayOfWeek', y='Sales', data=train, ax=axis1)

sns.barplot(x='DayOfWeek', y='Customers', data=train, ax=axis2) 
# Plot average sales and customers over months

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Month', y='Sales', data=train, ax=axis1)

sns.barplot(x='Month', y='Customers', data=train, ax=axis2)
# Plot average sales and customers with/without promo

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Promo', y='Sales', data=train, ax=axis1)

sns.barplot(x='Promo', y='Customers', data=train, ax=axis2)
# StateHoliday has values 0 & "0", So, we need to merge values with 0 to "0"

train["StateHoliday"].loc[train["StateHoliday"] == 0] = "0"



sns.countplot(x='StateHoliday',data=train)



# Plot average sales on StateHoliday

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='StateHoliday', y='Sales', data=train, ax=axis1)

filt = (train["StateHoliday"] != "0") & (train["Sales"] > 0)

sns.barplot(x='StateHoliday', y='Sales', data=train[filt], ax=axis2)
# Combining a,b and c type stores so as to reduce the bias

train["StateHoliday"] = train["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})

test["StateHoliday"]     = test["StateHoliday"].map({0: 0, "0": 0, "a": 1, "b": 1, "c": 1})



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='StateHoliday', y='Sales', data=train, ax=axis1)

sns.barplot(x='StateHoliday', y='Customers', data=train, ax=axis2)
# Visualizing Sales over SchoolHoliday

sns.countplot(x='SchoolHoliday',data=train)



# Plot average sales on StateHoliday

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='SchoolHoliday', y='Sales', data=train, ax=axis1)

sns.barplot(x='SchoolHoliday', y='Customers', data=train, ax=axis2)
train["Sales"].plot(kind='hist',bins=70,xlim=(0,15000))
store_df['PromoInterval'].value_counts()
store_df.head()
# Merging train and store_df

train_store = pd.merge(right=store_df,left=train,how='outer',on='Store')

train_store.head()
# Checking correlation bw different variables

plt.figure(figsize=(25,25))

sns.heatmap(train_store.corr(),vmax=.7,cbar=True,annot=True)
train_store.isnull().sum()
train_store["Sales_per_customer"].plot(kind='hist',bins=70,xlim=(0,25))
# Replacing missing values with median of their respective columns and dropping PromoInterval as it is not that much related to the dependent variable 



med_sales_per_customer = train_store['Sales_per_customer'].astype('float').median(axis=0)

train_store['Sales_per_customer'].replace(np.nan,med_sales_per_customer,inplace=True)



med_comp_dist = train_store['CompetitionDistance'].astype('float').median(axis=0)

train_store['CompetitionDistance'].replace(np.nan,math.floor(med_comp_dist),inplace=True)



med_comp_month = train_store['CompetitionOpenSinceMonth'].astype('float').median(axis=0)

train_store['CompetitionOpenSinceMonth'].replace(np.nan,math.floor(med_comp_month),inplace=True)



med_comp_year = train_store['CompetitionOpenSinceYear'].astype('float').median(axis=0)

train_store['CompetitionOpenSinceYear'].replace(np.nan,math.floor(med_comp_year),inplace=True)



med_promo2_week = train_store['Promo2SinceWeek'].astype('float').median(axis=0)

train_store['Promo2SinceWeek'].replace(np.nan,math.floor(med_promo2_week),inplace=True)



med_promo2_year = train_store['Promo2SinceYear'].astype('float').median(axis=0)

train_store['Promo2SinceYear'].replace(np.nan,math.floor(med_promo2_year),inplace=True)



train_store.drop(['PromoInterval'],axis=1,inplace=True)
sns.countplot(x='StoreType', data=train_store, order=['a','b','c', 'd'])



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='StoreType', y='Sales', data=train_store, order=['a','b','c', 'd'],ax=axis1)

sns.barplot(x='StoreType', y='Customers', data=train_store, order=['a','b','c', 'd'], ax=axis2)
sns.countplot(x='Assortment', data=train_store, order=['a','b','c'])



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Assortment', y='Sales', data=train_store, order=['a','b','c'], ax=axis1)

sns.barplot(x='Assortment', y='Customers', data=train_store, order=['a','b','c'], ax=axis2)
sns.countplot(x='Promo2', data=train_store)



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



sns.barplot(x='Promo2', y='Sales', data=train_store, ax=axis1)

sns.barplot(x='Promo2', y='Customers', data=train_store, ax=axis2)
sns.distplot(train_store['Customers'],color='Black')

train_store['Customers'].skew()
sns.distplot(train_store['Sales'],color='Black')

train_store['Sales'].skew()
train.drop(['Customers','Sales_per_customer'],axis=1,inplace=True)

train['Open'] = train['Open'].astype(float)
plt.figure(figsize=(15,15))

sns.heatmap(train.corr(),vmax=.7,cbar=True,annot=True)
train.drop(['Year','Month','SchoolHoliday','Date'],axis=1,inplace=True)

test.drop(['Year','Month','SchoolHoliday','Date'],axis=1,inplace=True)
# Create dummy varibales for DayOfWeek

train_dummies  = pd.get_dummies(train['DayOfWeek'], prefix='Day')

train_dummies.drop(['Day_7'], axis=1, inplace=True)



test_dummies = pd.get_dummies(test['DayOfWeek'],prefix='Day')

test_dummies.drop(['Day_7'], axis=1, inplace=True)



train = train.join(train_dummies)

test = test.join(test_dummies)
# fill NaN values in test with Open=1 if DayOfWeek != 7

test["Open"][test["Open"] != test["Open"]] = (test["DayOfWeek"] != 7).astype(float)
# Dropping DayOfWeek

train.drop(['DayOfWeek'], axis=1,inplace=True)

test.drop(['DayOfWeek'], axis=1,inplace=True)



# remove all rows(store,date) that were closed

train= train[train_store["Open"] != 0]
# Saving id's of those stores which were closed so we can put 0 in their respective sales column

closed_ids = test["Id"][test["Open"] == 0].values
# remove all rows(store,date) that were closed

test = test[test["Open"] != 0]
# Loop through each store, 

# train the model using the data of current store, and predict it's sales values.



train_dic = dict(list(train.groupby('Store')))

test_dic = dict(list(test.groupby('Store')))

submission = pd.Series()

scores = []



for i in test_dic:

    

    # current store

    store = train_dic[i]

    

    # define training and testing sets

    train_x = store.drop(["Sales","Store"],axis=1)

    train_y = store["Sales"]

    test_x = test_dic[i].copy()

    

    store_ids = test_x["Id"]

    test_x.drop(["Id","Store"], axis=1,inplace=True)

    

    # Linear Regression

    lr = lm.LinearRegression(normalize=True)

    lr.fit(train_x, train_y)

    pred_y = lr.predict(test_x)

    scores.append(lr.score(train_x, train_y))

    

    # append predicted values of current store to submission

    submission = submission.append(pd.Series(pred_y, index=store_ids))

    

# append rows(store,date) that were closed, and assign their sales value to 0

submission = submission.append(pd.Series(0, index=closed_ids))



# save to csv file

submission = pd.DataFrame({ "Id": submission.index, "Sales": submission.values})

submission.to_csv('rossmann.csv', index=False)