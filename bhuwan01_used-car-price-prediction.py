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
#importing needed lib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#read train and test data 
train = pd.read_csv('../input/used-cars-price-prediction/train-data.csv')
test = pd.read_csv('../input/used-cars-price-prediction/test-data.csv')

#concat both data

data = pd.concat([train,test],sort=False)
data.head()
#checking shape of data
data.shape

data.describe()
#Feature Engennering 
for i in range(data.shape[0]):
    data['company'][i] = data.iloc[i,1].split()[0]
    data['milege in kmpl'][i] = data.iloc[i,8].split()[0]
    data['engine in cc'][i] = data.iloc[i,9].split()[0]
    data['power in bhp'][i] = data.iloc[i,10].split()[0]
data.head()
# unwanted columns
removing_cols = ['Unnamed: 0','Name','Mileage','Engine','Power','New_Price']
#removing unwanted columns form data
data1 = data.drop(removing_cols,axis=1)
data1.head(100)
#replacing 'null' with average manually
data1['power in bhp'] = data1['power in bhp'].replace('null',90)
#for showing all rows and columns present in data
pd.set_option('display.max_row',None)
pd.set_option('display.max_columns',None)
#checking if null value is present or not
data1.isnull().sum()

data1['Seats']
#replacing nan values of seats manually
data1['Seats'] = data1['Seats'].replace(np.nan,6)
data1['Seats'] = data1['Seats'].replace(0.0,4)
#spliting the data into two part train and test data for visualization
train_data = data1[:6019]
train_data.shape
test_data = data1[6019:]
test_data.shape
#data visualization
sns.distplot(train_data['Price'])
plt.figure(figsize=(20,20))
sns.barplot(x='Year',y='Price',data=train_data)
plt.figure(figsize=(10,10))
sns.barplot(x='Location',y='Price',data=train_data)
plt.figure(figsize=(20,20))
sns.barplot(x='Seats',y='Price',data=train_data)
plt.figure(figsize=(20,20))
sns.barplot(x='company',y='Price',data=train_data)
plt.figure(figsize=(20,20))
sns.barplot(x='Fuel_Type',y='Price',data=train_data)
#changing the data type from string to float
data1['milege in kmpl'] = data1['milege in kmpl'].astype(float)
data1['engine in cc'] = data1['engine in cc'].astype(float)
data1['power in bhp'] = data1['power in bhp'].astype(float)
#checking for categorical variable
cat_var = data1.select_dtypes(include=['object']).columns
#working with categorical variable/data
data2 = pd.get_dummies(data1[cat_var],drop_first=True)
data3 = pd.concat([data1,data2],axis=1)
data4 = data3.drop(data3[cat_var],axis=1)
#spliting data into two part train and test for machine learning after working on categorial variable/data
X_trains = data4[:6019]
X_train = X_trains.drop('Price',axis=1)
X_train.shape
X_test = data4[6019:]
X_test = X_test.drop('Price',axis=1)
X_test.shape
y_train = X_trains['Price']
#I tried many algorithms but this perform best
from sklearn.ensemble import ExtraTreesRegressor
etr= ExtraTreesRegressor()
etr.fit(X_train,y_train)
p = etr.predict(X_test)
print('accuracy_on_training:',etr.score(X_train,y_train))
