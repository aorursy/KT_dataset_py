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
train_df=pd.read_csv('../input/szeged-weather/weatherHistory.csv')
train_df.head()
del train_df["Loud Cover"]

import datetime
      

train_df["Date"]=pd.to_datetime(train_df["Formatted Date"],utc=True)
train_df["Month"]=train_df["Date"].dt.month

train_df
del train_df["Formatted Date"]
del train_df["Daily Summary"]
del train_df["Date"]
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder()  
train_df['Summary']= label_encoder.fit_transform(train_df['Summary'])   
train_df['Summary'].unique() 
train_df["Precip Type"]=train_df["Precip Type"].replace(['rain','snow'],[0,1])
train_df[train_df.columns[:]].corr()["Apparent Temperature (C)"][:]
train_df.isna().sum()
import numpy as np
train_df["Precip Type"]=train_df["Precip Type"].replace(np.nan,0)
train_df.isna().sum()
del train_df["Pressure (millibars)"]
for i in train_df.columns.values:
    print(i+"    "+str(train_df[i].skew()))
train_df
target=train_df["Apparent Temperature (C)"]
del train_df["Apparent Temperature (C)"]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train,X_test,y_train,y_test=train_test_split(train_df,target,random_state=0)

reg = LinearRegression()
reg.fit(X_train,y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(y_test,reg.predict(X_test)))
print(rmse)

print(reg.score(X_test,y_test))
print(reg.score(X_train,y_train))
