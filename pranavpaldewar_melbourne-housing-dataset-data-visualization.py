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
data = pd.read_csv('../input/california-housing-prices/housing.csv')
data.info()
data.corr()
data.head()
data.describe()
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.scatter('latitude','longitude',data=data,alpha=0.1)
plt.xlabel('latitude')
plt.ylabel('longitude')

plt.figure(figsize=(10,6))
plt.scatter('latitude','longitude',data=data,alpha=0.2,c=data.median_house_value,cmap=plt.get_cmap('jet'))
plt.xlabel('latitude')
plt.ylabel('longitude')
#sacatter plot
plt.figure(num=1,figsize=(10,8))
plt.scatter('median_income','median_house_value',data=data,alpha=0.1)
plt.xlabel('median_income')
plt.ylabel('median_house_value')
plt.figure(num=2,figsize=(10,8))
plt.scatter('housing_median_age','median_house_value',data=data,alpha=0.1)
plt.xlabel('housing_median_age')
plt.ylabel('median_house_value')
import seaborn as sns
plt.figure(num=3,figsize=(10,9))
sns.countplot(data[data['housing_median_age']>30]['housing_median_age'])
plt.figure(num=4,figsize=(10,9))
sns.countplot(data[data['housing_median_age']<=30]['housing_median_age'])
plt.figure(num=5,figsize=(10,9))
sns.countplot(data[data['housing_median_age']>30]['housing_median_age'])
data.head()


data.hist(bins=50
          ,figsize=(20,15))
sns.distplot(data.median_income)
sns.boxplot(data.median_income)
data1 = data.copy()
print(data.ocean_proximity.unique())
print(data.ocean_proximity.value_counts())
data.ocean_proximity.value_counts().plot(kind='bar')
data1['rooms_per_household']=data.total_rooms/data.households

print(data[data['ocean_proximity']=='<1H OCEAN']['median_house_value'].median())
print(data[data['ocean_proximity']=='NEAR BAY']['median_house_value'].median())
print(data[data['ocean_proximity']=='INLAND']['median_house_value'].median())



from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
new_data =label.fit_transform(data1.ocean_proximity)
data1['ocean_proximity']=new_data
data1.total_bedrooms=data.total_bedrooms.fillna(0)
data1.ocean_proximity
data.head()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
lr= LinearRegression(normalize=True)
col = data1.columns
y = data1.median_house_value
x = data1.drop('median_house_value',axis=1)
train_x,val_x,train_y,val_y=train_test_split(x,y,train_size=0.8)
lr.fit(train_x,train_y)
predictions = lr.predict(val_x)
print("Mean sqaured Error={}".format(mse(predictions,val_y)))

from sklearn.ensemble import RandomForestRegressor as rfs
def n_estim(n):
    d={}
    for n1 in n:
        model_1 =rfs(n_estimators=n1)
        model_1.fit(train_x,train_y)
        predictions_2=model_1.predict(val_x)
        d[n1]=mse(predictions_2,val_y)
    return d
l =[50,100,150,200]
print(n_estim(l))