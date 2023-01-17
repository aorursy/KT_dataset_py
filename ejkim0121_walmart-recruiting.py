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
train=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv")

train.head()
test=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv")

test.head()
y=train["Weekly_Sales"]
train.shape
train.dtypes
train["Date"]=train["Date"].astype("datetime64")

test["Date"]=test["Date"].astype("datetime64")

train.dtypes
#train["day of week"]=train["Date"].dt.dayofweek

#test["day of week"]=test["Date"].dt.dayofweek

train["month"]=train["Date"].dt.month

test["month"]=test["Date"].dt.month

train["day"]=train["Date"].dt.day

test["day"]=test["Date"].dt.day

train["year"]=train["Date"].dt.year

test["year"]=test["Date"].dt.year
#train["day of week"].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

A,B=plt.subplots(1,1,figsize=(10,10))

sns.boxplot(train["month"],train["Weekly_Sales"])
store=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv")

store.head()
train=pd.merge(train,store)

test=pd.merge(test,store)
train["Type"].unique()
train["Type"]=train["Type"].replace({"A":1,"B":2,"C":3})

test["Type"]=test["Type"].replace({"A":1,"B":2,"C":3})
train=train.drop(["Weekly_Sales","Date"],axis=1)

train.head()
tesst=test.drop(["Date"],axis=1)

test.head()
#from lightgbm import LGBMRegressor 

#lgb=LGBMRegressor()

#lgb.fit(train,y)



from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=100, n_jobs=-1)

rf.fit(train,y)
value=rf.predict(test)
sample=pd.read_csv("/kaggle/input/walmart-recruiting-store-sales-forecasting/sampleSubmission.csv")

sample.head()
sample["Weekly_Sales"]=value
sample.head()
sample.to_csv("20191014.csv",index=False)