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
df = pd.read_csv('/kaggle/input/delhi-house-price-prediction/MagicBricks.csv')
df
df.isnull().sum()
df.Bathroom.value_counts()
df.Furnishing.value_counts()
df.Parking.value_counts()
df.Type.value_counts()
df.Per_Sqft.value_counts()
df.Bathroom.fillna(2.0,inplace=True)
df.Furnishing.fillna('Semi-Furnished',inplace=True)

df.Parking.fillna(1.0,inplace=True)

df.Type.fillna('Builder_Floor',inplace=True)

df.Per_Sqft.fillna(12500.0,inplace=True)
df.info()
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

df['Furnishing']=label.fit_transform(df.Furnishing)

df['Locality']=label.fit_transform(df.Locality)

df['Status']=label.fit_transform(df.Status)

df['Transaction']=label.fit_transform(df.Transaction)

df['Type']=label.fit_transform(df.Type)
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=0.1,random_state=1)

def data_split(df):

    x = df.drop('Price',axis=1)

    y = df['Price']

    return x,y

x_train,y_train=data_split(train)

x_test,y_test=data_split(test)
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LinearRegression 

log = LinearRegression()           

log.fit(x_train,y_train) 

log_train = log.score(x_train,y_train)  

log_test = log.score(x_test,y_test)

print(log_train)                                                                                           

print(log_test)

from sklearn.ensemble import RandomForestRegressor

regress = RandomForestRegressor()

regress.fit(x_train , y_train)

reg_train = regress.score(x_train , y_train)

reg_test = regress.score(x_test , y_test)

print(reg_train)                                                                                   

print(reg_test)

import seaborn as sns
sns.countplot(df['BHK'])
sns.countplot(df['Bathroom'])