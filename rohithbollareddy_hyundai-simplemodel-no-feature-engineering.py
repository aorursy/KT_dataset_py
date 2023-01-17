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
import matplotlib.pyplot as plt 

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

lr=LinearRegression()

xgb=XGBRegressor()
data=pd.read_csv('../input/used-car-dataset-ford-and-mercedes/hyundi.csv')

data.head()
data=data[['model','year','transmission','mileage','fuelType','tax(£)','mpg','engineSize','price']]

data.head()
data.info()
data.describe()
data.isnull().any()
data['model'].unique()
data['transmission'].unique()
data['fuelType'].unique()
corr_=data.corr()

sns.heatmap(corr_,annot=True)
plt.figure(figsize=(10,6))

plt.scatter(data['engineSize'],data['price'])

plt.xlabel("Engine Size")

plt.ylabel('Price')

plt.title("Engine Size Vs Price")
plt.figure(figsize=(15,10))

sns.countplot(data['model'])
data_ser_=data.model.value_counts()

data_ser=pd.DataFrame(data_ser_)

labels=data_ser.index

sizes=data_ser['model']
fig1,ax1=plt.subplots()

ax1.pie(sizes,explode=None,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)

ax1.axis('equal')

plt.show()
sns.countplot(data['fuelType'])

plt.title('Fuel Type')
sns.countplot(data['transmission'])

plt.title('Transmission Type')
sns.boxplot(data['mileage'])
sns.boxplot(data['engineSize'])
sns.boxplot(data['price'])
sns.boxplot(data['mpg'])
data['model']=le.fit_transform(data['model'])

data['transmission']=le.fit_transform(data['transmission'])

data['fuelType']=le.fit_transform(data['fuelType'])
X=data[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax(£)', 'mpg','engineSize']]

Y=data[['price']]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
lr.fit(X_train,Y_train)

lr_pred=lr.predict(X_test)

print(lr.score(X_test,Y_test))
xgb.fit(X_train,Y_train)
xgb_pred=xgb.predict(X_test)
xgb.score(X_test,Y_test)