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
df=pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')






df.head()
df.shape
df.isnull().any()
df.describe()
df['price'].max()
df.info()
df['fuelType'].unique()
df['transmission'].unique()
df['fuelType'].replace('Petrol',0,inplace=True)
df['fuelType'].replace('Diesel',1,inplace=True)
df['fuelType'].replace('Hybrid',2,inplace=True)
df['fuelType']=df['fuelType'].astype(int)
df.dtypes
df['transmission'].replace('Manual',0,inplace=True)
df['transmission'].replace('Automatic',1,inplace=True)
df['transmission'].replace('Semi-Auto',2,inplace=True)
df['transmission']=df['transmission'].astype(int)

df.dtypes
cor=df.corr()
cor
import seaborn as sns
sns.heatmap(cor)
t=abs(cor['price'])
colls=t[t>0.2]
colls
import matplotlib.pyplot as plt
%matplotlib inline
sns.barplot(x='transmission',y='price',data=df)
sns.scatterplot(x='mileage',y='price',data=df)
sns.scatterplot(x='engineSize',y='mpg',data=df)
feature_cols=['year','transmission','mileage','tax','mpg','engineSize']
x=df[feature_cols]
y=df.price
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
from sklearn import svm
sv=svm.SVR()
sv.fit(x_train,y_train)
pred=sv.predict(x_test)
pred
from sklearn.metrics import accuracy_score
s=sv.score(x,y)
s
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
tree.fit(x_train,y_train)
p=tree.predict(x_test)
p
s=tree.score(x,y)
print('accuracy score',s)
