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
train=pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

train.head(10)
test=pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")

test.head(10)
train.dtypes
train.shape
train["datetime"]=train["datetime"].astype("datetime64")

test["datetime"]=test["datetime"].astype("datetime64")

train.dtypes
train["hour"]=train["datetime"].dt.hour

test["hour"]=test["datetime"].dt.hour

train["year"]=train["datetime"].dt.year

test["year"]=test["datetime"].dt.year

train["day of week"]=train["datetime"].dt.dayofweek

test["day of week"]=test["datetime"].dt.dayofweek
import matplotlib.pyplot as plt

import seaborn as sns

E,J=plt.subplots(2,1,figsize=(20,12))

sns.distplot(train["count"],ax=J[0])

sns.distplot(np.log(train["count"]),ax=J[1])



#E,J=plt.subplots(1,1,figsize=(20,12))

#sns.boxplot(train["hour"],train["count"])



#E,J=plt.subplots(1,1,figsize=(20,12))

#sns.boxplot(train["year"],train["count"])
y=np.log(train["count"])

y.head()
train=train.drop(["datetime","casual","registered","count"],1)

train.head()
test=test.drop(["datetime"],1)

test.head()
# from sklearn.ensemble import RandomForestRegressor

# rf=RandomForestRegressor(n_estimators=100,random_state=0)

# rf.fit(train,y)
#rf.predict(test)
from lightgbm import LGBMRegressor

lgb=LGBMRegressor()

lgb.fit(train,y)
sub=pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")

sub.head()
# sub["count"]=np.exp(rf.predict(test)

# sub.head()

sub["count"]=np.exp(lgb.predict(test))

sub.head()
sub.to_csv("20190921.csv",index=False)