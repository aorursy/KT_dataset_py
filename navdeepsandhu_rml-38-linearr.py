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
train = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv',encoding='Latin-1')

#this is just a comment

train.head(5)
train.describe()

# this explains the data
train.dtypes

#this gives the type of data in the column dtypes=data type
#Step 3: Clean up data

# Use the .isnull() method to locate missing data

missing_values = train.isnull()

missing_values.tail(5)
#Step 4.1: Visualize the data

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

%matplotlib inline
train_data,test_data=train_test_split(train,train_size=0.7,random_state=100)

reg=linear_model.LinearRegression()

x_train=np.array(train_data['price']).reshape(-1,1)

y_train=np.array(train_data['sqft_living']).reshape(-1,1)

reg.fit(x_train,y_train)



x_test=np.array(test_data['price']).reshape(-1,1)

y_test=np.array(test_data['sqft_living']).reshape(-1,1)

pred=reg.predict(x_test)

print('linear model')

mean_squared_error=metrics.mean_squared_error(y_test,pred)

print('Sqaured mean error', round(np.sqrt(mean_squared_error),2))

print('R squared training',round(reg.score(x_train,y_train),3))

print('R sqaured testing',round(reg.score(x_test,y_test),3) )

print('intercept',reg.intercept_)

print('coefficient',reg.coef_)
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(15,10))

plt.title("house prices by sqft_living")

plt.xlabel('floors')

plt.ylabel('sqft_living')

plt.legend()

sns.barplot(x='floors',y='sqft_living',data=train)
train_data,test_data=train_test_split(train,train_size=0.7,random_state=100)

reg=linear_model.LinearRegression()

x_train=np.array(train_data['floors']).reshape(-1,1)

y_train=np.array(train_data['sqft_living']).reshape(-1,1)

reg.fit(x_train,y_train)



x_test=np.array(test_data['floors']).reshape(-1,1)

y_test=np.array(test_data['sqft_living']).reshape(-1,1)

pred=reg.predict(x_test)

print('linear model')

mean_squared_error=metrics.mean_squared_error(y_test,pred)

print('Sqaured mean error', round(np.sqrt(mean_squared_error),2))

print('R squared training',round(reg.score(x_train,y_train),3))

print('R sqaured testing',round(reg.score(x_test,y_test),3) )

print('intercept',reg.intercept_)

print('coefficient',reg.coef_)