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
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df1=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df.head()
df.shape
df1.head()
df.isnull().any()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.scatterplot(x=df['GRE Score'],y=df['Research'])
cor=df.corr()
cor
sns.heatmap(cor)
sns.boxplot(df['GRE Score'])
df.info()
df.describe()
sns.jointplot(df['GRE Score'],df['University Rating'])
sns.distplot(df['GRE Score'])
sns.distplot(df['Research'],kde=False)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=df.iloc[:,[1,2,3,4,5,6,7]]
y=df.iloc[:,[8]]
# as the size of data set is small iam not dividing into train and test
x=sc.fit_transform(x)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)
pred=lr.predict(x)
pred
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y,pred))
sns.distplot(pred-y,color='r')
from sklearn.metrics import r2_score
print(r2_score(y,pred))
from sklearn.tree import DecisionTreeRegressor
tr=DecisionTreeRegressor()
tr.fit(x,y)
pre=tr.predict(x)
pre
print(mean_squared_error(y,pre))
print(r2_score(y,pre))
import xgboost
xgb=xgboost.XGBRegressor()
xgb.fit(x,y)
pr=xgb.predict(x)
pr
print(mean_squared_error(y,pr))
print(r2_score(y,pr))
print('MSE and r2_score for linear regression are',mean_squared_error(pred,y),'and',r2_score(pred,y))
print('MSE and r2_score for decision tree  are',mean_squared_error(pre,y),'and',r2_score(pre,y))
print('MSE and r2_score for xgboost model are',mean_squared_error(pr,y),'and',r2_score(pr,y))

