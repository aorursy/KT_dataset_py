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
data=pd.read_csv('../input/autompg-dataset/auto-mpg.csv')
data.head()
data.drop('car name',axis=1,inplace=True)
data['horsepower'].value_counts()
data['horsepower'].unique()
data['horsepower']=data['horsepower'].replace('?','150')
data['horsepower']=data['horsepower'].astype('int')
'?' in data['horsepower']
y=pd.DataFrame()
y['mpg']=data['mpg']
data.drop('mpg',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(data,y,random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train=pd.DataFrame(scaler.transform(X_train))
X_test=pd.DataFrame(scaler.transform(X_test))

scaler2=MinMaxScaler()
scaler2.fit(y_train)
y_train=pd.DataFrame(scaler2.transform(y_train))
y_test=pd.DataFrame(scaler2.transform(y_test))
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2,include_bias=False)
poly.fit(X_train)
X_train=pd.DataFrame(poly.transform(X_train))
X_test=pd.DataFrame(poly.transform(X_test))
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
print(reg.score(X_train,y_train))
print(reg.score(X_test,y_test))