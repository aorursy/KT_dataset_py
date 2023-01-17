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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Loading Data

train_data = pd.read_csv("/kaggle/input/big-mart-sales-prediction/Train.csv")

test_data = pd.read_csv("/kaggle/input/big-mart-sales-prediction/Test.csv")
train_data.head()
test_data.head()
train_data.describe()
train_data.info()
train_data.shape

test_data.shape
train_data.isnull().sum()
test_data.isnull().sum()
train_data.apply(lambda x: len(x.unique()))
# Handling Missing Values 

train_data['Item_Weight'].fillna(train_data['Item_Weight'].mean(),inplace=True)

test_data['Item_Weight'].fillna(test_data['Item_Weight'].mean(),inplace=True)



train_data['Outlet_Size'].fillna(train_data['Outlet_Size'].mode()[0],inplace=True)

test_data['Outlet_Size'].fillna(test_data['Outlet_Size'].mode()[0],inplace=True)
train_data.dtypes
train_data["Item_Fat_Content"].value_counts()
# Here we an see that Low Fat and LF are same thing and reg and regilar is same thing 

# Replacing LF and low fat  with Low Fat

train_data.Item_Fat_Content.replace('LF','Low Fat',inplace=True)

train_data.Item_Fat_Content.replace('low fat','Low Fat',inplace=True)

test_data.Item_Fat_Content.replace('LF','Low Fat',inplace=True)

test_data.Item_Fat_Content.replace('low fat','Low Fat',inplace=True)

# Replacing reg with Regular 

train_data.Item_Fat_Content.replace('reg','Regular',inplace=True)

test_data.Item_Fat_Content.replace('reg','Regular',inplace=True)
train_data["Item_Fat_Content"].value_counts()
train_data.Item_Type.value_counts().plot.bar(color='Red',figsize=(12,8))
sns.heatmap(train_data.corr(),vmin=-1,vmax=3,square=True,annot=True,cmap='RdYlGn')
train_data.dtypes
train_data.Outlet_Identifier.value_counts()
train_data.Item_Fat_Content.value_counts()
train_data.Outlet_Size.value_counts()
train_data.Outlet_Location_Type.value_counts()
train_data.Outlet_Type.value_counts()
train_data.Item_Type.value_counts()
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

# For Train Data

train_data['Outlet_Identifier']=le.fit_transform (train_data['Outlet_Identifier'])

train_data['Item_Fat_Content']=le.fit_transform (train_data['Item_Fat_Content'])

train_data['Outlet_Size']=le.fit_transform (train_data['Outlet_Size'])

train_data['Outlet_Location_Type']=le.fit_transform (train_data['Outlet_Location_Type'])

train_data['Outlet_Type']=le.fit_transform (train_data['Outlet_Type'])

train_data['Item_Type']=le.fit_transform (train_data['Item_Type'])



# For Test Data

test_data['Outlet_Identifier']=le.fit_transform (test_data['Outlet_Identifier'])

test_data['Item_Fat_Content']=le.fit_transform (test_data['Item_Fat_Content'])

test_data['Outlet_Size']=le.fit_transform (test_data['Outlet_Size'])

test_data['Outlet_Location_Type']=le.fit_transform (test_data['Outlet_Location_Type'])

test_data['Outlet_Type']=le.fit_transform (test_data['Outlet_Type'])

test_data['Item_Type']=le.fit_transform (test_data['Item_Type'])

train_data.head()
train_data.info()
test_data.info()
train = train_data.drop(['Item_Identifier','Item_Visibility'], axis=1)

test = test_data.drop(['Item_Identifier','Item_Visibility'], axis=1)



train.head()
y = train['Item_Outlet_Sales']

x = train.drop('Item_Outlet_Sales', axis = 1)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

X_train, X_test, Y_train, Y_test = train_test_split(x,y, random_state = 0, test_size = 0.25)

model = RandomForestRegressor(random_state=40)

model.fit(X_train, Y_train)

pred_y = model.predict(X_test)



model.score(X_train,Y_train)
sns.distplot(pred_y,hist=False)

sns.distplot(Y_test,hist=False,color='r')