import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/usa-housing-dataset/housing_train.csv')
data.head()
data.describe()
data.info()
plt.figure(figsize=(10,7))

sns.heatmap(data.isna())
nullCols = []



for i in data.columns:

    if(data[i].isna().sum()>=146):

        data.drop(i,inplace=True,axis=1)

    else:

        data[i] = data[i].fillna(method='ffill')
sns.heatmap(data.isna())
len(data.columns)
data.duplicated().sum()
data.info()
plt.figure(figsize=(15,10))

sns.heatmap(data.corr(),annot=True,fmt='.1f')
sns.countplot(data=data,x='SaleCondition')
dt = data.dtypes==object

col = data.columns[dt].tolist()
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()

x = data[col].apply(le.fit_transform)
oneList=[]

one = OneHotEncoder()

oneData = one.fit_transform(x).toarray()
oneData.shape
data.drop('Id',inplace=True,axis=1)

dt = data.dtypes!=object

col = data.columns[dt].tolist()

new_x = np.append(data[col],x,axis=1)

new_x.shape
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
train_x,test_x,train_y,test_y = train_test_split(new_x,data['SalePrice'],test_size=0.2,random_state=101)
sc = StandardScaler()

train_x = sc.fit_transform(train_x)

test_x = sc.transform(test_x)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,mean_squared_error
lr = LinearRegression()

lr.fit(train_x,train_y)

predict = lr.predict(test_x)

print('MAE ',mean_absolute_error(test_y,predict))

print('MSE ',mean_squared_error(test_y,predict))

print('RMSE ',np.sqrt(mean_squared_error(test_y,predict)))