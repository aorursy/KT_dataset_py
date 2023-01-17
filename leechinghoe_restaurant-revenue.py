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
!pip install seaborn --upgrade
#import libraries

import datetime as dt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.__version__
train = pd.read_csv("../input/restaurant-revenue-prediction/train.csv.zip",index_col="Id")
test = pd.read_csv("../input/restaurant-revenue-prediction/test.csv.zip",index_col="Id")
print("train set shape : ",train.shape)
print("test set shape : ",test.shape)
print("train set number of null value : ",train.isnull().sum().sum())
print("test set number of null value : ",test.isnull().sum().sum())
print("No missing value in both dataset")
train.describe()
test.describe()
train.columns
train.select_dtypes(exclude="number")
train["City"].unique()
train.City.nunique()
train["Type"].unique()
train["City Group"].unique()
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
plt.title("Restaurant Type and Revenue")
sns.boxplot(y="revenue",x="Type",data=train)
train.groupby(by="Type").revenue.median()
train.groupby(by="Type").revenue.mean()
train.Type.value_counts()
test.Type.value_counts()
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title(label="Test Set City Group Count")
sns.countplot(x="Type", data=test)
plt.subplot(2,2,2)
plt.title(label="Train Set City Group Count")
sns.countplot(x="Type", data=train)
print("Train Set Type Inline Percentage : ",100*len(train.loc[train.Type=="IL"])/len(train),"%")
print("Train Set Type Food Court Percentage : ",100*len(train.loc[train.Type=="FC"])/len(train),"%")
print("Train Set Type Drive Thru Percentage : ",100*len(train.loc[train.Type=="DT"])/len(train),"%")
print("Test Set Type Inline Percentage : ",100*len(test.loc[test.Type=="IL"])/len(test),"%")
print("Test Set Type Food Court Percentage : ",100*len(test.loc[test.Type=="FC"])/len(test),"%")
print("Test Set Type Drive Thru Percentage : ",100*len(test.loc[test.Type=="DT"])/len(test),"%")
print("Test Set Type Mobile Percentage : ",100*len(test.loc[test.Type=="MB"])/len(test),"%")
#move to bivariate later
# sns.set_style("whitegrid")
# plt.figure(figsize=(16, 6))
# plt.title("City Group and Revenue")
# sns.boxplot(y="revenue",x="City Group",data=train)
train.groupby(by="City Group").revenue.median()
train.groupby(by="City Group").revenue.mean()
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.title(label="Test Set City Group Count")
sns.countplot(x="City Group", data=test)
plt.subplot(2,2,2)
plt.title(label="Train Set City Group Count")
sns.countplot(x="City Group", data=train)
print("Train Set Big Cities Percentage : ",100*len(train.loc[train["City Group"]=="Big Cities"])/len(train),"%")
print("Train Set Other Percentage : ",100*len(train.loc[train["City Group"]=="Other"])/len(train),"%")
print("Test Set Big Cities Percentage : ",100*len(test.loc[test["City Group"]=="Big Cities"])/len(test),"%")
print("Test Set Other Percentage : ",100*len(test.loc[test["City Group"]=="Other"])/len(test),"%")
#move to bivariate later
# plt.figure(figsize=(40, 6))
# plt.title(label="City and Revenue")
# sns.boxplot(y="revenue",x="City",data=train)
train.groupby(by="City").revenue.median()
train.City.unique()
print(len(train.City.unique()))
print(test.City.unique())
print(len(test.City.unique()))
solo_set_city=[]
for i in test.City.unique().tolist()+train.City.unique().tolist():
    if i not in train.City.unique() or i not in test.City.unique():
         solo_set_city.append(i)
print(solo_set_city)
print(len(solo_set_city))
plt.figure(figsize=(40, 6))
plt.title(label="Train Set City Count")
sns.countplot(x="City", data=train)
for city in train.City.unique():
    print("Train Set "+str(city)+" Percentage : ",100*len(train.loc[train["City"]==city])/len(train),"%")
plt.figure(figsize=(40, 6))
plt.title(label="Test Set City Count")
sns.countplot(x="City", data=test)
for city in test.City.unique():
    print("Test Set "+str(city)+" Percentage : ",100*len(test.loc[test["City"]==city])/len(test),"%")
num_cols = train.select_dtypes(include="number").drop('revenue',axis=1).columns
len(num_cols)
train.P1.describe()
# f, axes = plt.subplots(13, 6,figsize=(40,40))

# for j in range(len(axes)):
#     count=0
#     for i in range(0,3):
#         if j*3+i >=37:
#             break
#         sns.violinplot(  y=num_cols[j*3+i], data=train ,ax=axes[j,i+count],cut=0)
#         axes[j,i+count].set_title(num_cols[j*3+i]+"Train")
#         axes[j,i+count].set_ylim(min(test[num_cols[j*3+i]].min(),train[num_cols[j*3+i]].min()), max(test[num_cols[j*3+i]].max(),train[num_cols[j*3+i]].max()))
#         count+=1
#         sns.violinplot(  y=num_cols[j*3+i], data=test,ax=axes[j,i+count],cut=0)
#         axes[j,i+count].set_title(num_cols[j*3+i]+"Test")
#         axes[j,i+count].set_ylim(min(test[num_cols[j*3+i]].min(),train[num_cols[j*3+i]].min()), max(test[num_cols[j*3+i]].max(),train[num_cols[j*3+i]].max()))
    
#checking
sns.violinplot(y="P25",data=train,cut=0)
test.P3.unique()
for i in num_cols:
    print(train[i].value_counts())
for i in num_cols:
    print(test[i].value_counts())
#plotting histogram of same features with train and test set back to back
# f, axes = plt.subplots(13, 6,figsize=(60,80))
# for j in range(len(axes)):
#     count=0
#     for i in range(0,3):
#         if j*3+i >= 37:
#             break
#         sns.histplot(  x=num_cols[j*3+i], data=train ,ax=axes[j,i+count], discrete=True)
#         axes[j,i+count].set_title(num_cols[j*3+i]+"Train")
#         axes[j,i+count].set_xlim(min(test[num_cols[j*3+i]].min(),train[num_cols[j*3+i]].min()), max(test[num_cols[j*3+i]].max(),train[num_cols[j*3+i]].max()))
#         count+=1
#         sns.histplot(  x=num_cols[j*3+i], data=test,ax=axes[j,i+count] ,discrete=True)
#         axes[j,i+count].set_title(num_cols[j*3+i]+"Test")
#         axes[j,i+count].set_xlim(min(test[num_cols[j*3+i]].min(),train[num_cols[j*3+i]].min()), max(test[num_cols[j*3+i]].max(),train[num_cols[j*3+i]].max()))
# plt.tight_layout()
# plt.show()
train["open_dt"]=pd.to_datetime(train['Open Date'])
test["open_dt"]=pd.to_datetime(test['Open Date'])
train.drop('Open Date',axis=1,inplace=True)
test.drop('Open Date',axis=1,inplace=True)
f, axes = plt.subplots(1, 2,figsize=(12,5))
sns.histplot(x="open_dt",data=train,ax=axes[0],bins=12)
sns.histplot(x="open_dt",data=test,ax=axes[1],bins=12)
train['year']=pd.DatetimeIndex(train['open_dt']).year
test['year']=pd.DatetimeIndex(test['open_dt']).year
f, axes = plt.subplots(1, 2,figsize=(40,10))
sns.histplot(x="year",data=train,ax=axes[0],discrete=True)
axes[0].set_xlim(1995,2014)
axes[0].set_xticks(range(1995,2015))

sns.histplot(x="year",data=test,ax=axes[1],discrete=True)
axes[1].set_xlim(1995,2014)
axes[1].set_xticks(range(1995,2015))
#train set distribution
for year in sorted(train.year.unique(),reverse=True):
    print("Train Set "+str(year)+" Percentage : ",100*len(train.loc[train["year"]==year])/len(train),"%")
    check+=100*len(train.loc[train["year"]==year])/len(train)
#test set distribution
for year in sorted(test.year.unique(),reverse=True):
    print("Test Set "+str(year)+" Percentage : ",100*len(test.loc[test["year"]==year])/len(test),"%")
    summa+=100*len(test.loc[test["year"]==year])/len(test)
train['day']=pd.DatetimeIndex(train['open_dt']).day
test['day']=pd.DatetimeIndex(test['open_dt']).day
f, axes = plt.subplots(1, 2,figsize=(40,10))
sns.histplot(x="day",data=train,ax=axes[0],discrete=True)
axes[0].set_xlim(1,31)
axes[0].set_xticks(range(0,33))

sns.histplot(x="day",data=test,ax=axes[1],discrete=True)
axes[1].set_xlim(1,31)
axes[1].set_xticks(range(0,33))
for day in sorted(test.day.unique(),reverse=True):
    print("Train Set "+str(day)+" Percentage : ",100*len(train.loc[train["day"]==day])/len(train),"%")
    print("Test Set "+str(day)+" Percentage : ",100*len(test.loc[test["day"]==day])/len(test),"%")
train["month"]=pd.DatetimeIndex(train['open_dt']).month
test['month']=pd.DatetimeIndex(test['open_dt']).month
f, axes = plt.subplots(1, 2,figsize=(40,10))
sns.histplot(x="month",data=train,ax=axes[0],discrete=True)
axes[0].set_xlim(1,12)
axes[0].set_xticks(range(0,14))

sns.histplot(x="month",data=test,ax=axes[1],discrete=True)
axes[1].set_xlim(1,12)
axes[1].set_xticks(range(0,14))
for month in sorted(test.month.unique(),reverse=True):
    print("Train Set "+str(month)+" Percentage : ",100*len(train.loc[train["month"]==month])/len(train),"%")
    print("Test Set "+str(month)+" Percentage : ",100*len(test.loc[test["month"]==month])/len(test),"%")
train.columns
cols=list(train.columns)
cols.remove('revenue')
cols=cols[3:]+cols[0:3]
cols.append('revenue')
cols
train
train=train[cols]
train.columns
corr=train.corr()
plt.figure(figsize=(40,20))
sns.heatmap(corr,annot=True)
test.columns
test.columns
test_cols=list(test.columns)[3:]+list(test.columns)[0:3]
len(test_cols)
test=test[test_cols]
corr=test.corr()
plt.figure(figsize=(40,20))
sns.heatmap(corr,annot=True)
ax = sns.barplot(x="City Group", y="revenue", hue="Type", data=train)
train.select_dtypes(exclude='number')
