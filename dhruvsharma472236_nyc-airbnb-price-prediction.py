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
data=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#head of dataset
data.head()
#info of the dataset
data.info()
#describing dataset
data.describe()
#shape of dataset
data.shape
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

sns.barplot(y='price',x='neighbourhood_group',data=data)
pivot=data.pivot_table(values='price', index='neighbourhood_group', columns='room_type')
pivot
bool=data.isna()
bool.head()
sns.heatmap(bool,yticklabels=False)
data['reviews_per_month'].fillna(0,inplace=True)
sns.heatmap(data.isna(),yticklabels=False)
sns.heatmap(pivot,linecolor='white',linewidth=0.3)
sns.barplot(y='price',x='room_type',data=data)
sns.barplot(y='number_of_reviews',x='neighbourhood_group',data=data)
sns.barplot(y='minimum_nights',x='neighbourhood_group',data=data)
rooms=pd.get_dummies(data['room_type'])
rooms.head()
city=pd.get_dummies(data['neighbourhood_group'])
city.head()
data=pd.concat([data,city,rooms],axis=1)
data.head()
data.drop(['id','name','host_id','host_name','neighbourhood_group','room_type', 'last_review'],axis=1,inplace=True)
data.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=data.drop(['neighbourhood','latitude','longitude','price','reviews_per_month'],axis=1)
y=data['price']
#now splitting data into train and test model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
model=LinearRegression()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
error=pd.DataFrame(np.array(y_test).flatten(),columns=['actual'])
error['prediction']=np.array(predictions)
error.head(10)