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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as nm
data=pd.read_csv("../input/used-car-dataset-ford-and-mercedes/audi.csv")
data.head()
data.info()
data.describe()
data.isnull().sum()
data['model'].unique()
data['year'].unique()
data['transmission'].unique()
data['transmission'].replace('Manual',1,inplace=True)
data['transmission']
data['transmission'].replace('Automatic',2,inplace=True)
data['transmission'].replace('Semi-Auto',3,inplace=True)
data['transmission']
data['mileage'].unique()
data['fuelType'].unique()
data['fuelType'].replace('Petrol',1,inplace=True)
data['fuelType'].replace('Diesel',2,inplace=True)
data['fuelType'].replace('Hybrid',3,inplace=True)
data['fuelType']
data['tax'].unique()
data['mpg'].unique()
data['engineSize'].unique()
sns.barplot(x='transmission',y='price',data=data)
sns.scatterplot(x='mileage',y='price',data=data)
sns.jointplot(x='tax',y='price',data=data)
sns.scatterplot(x='year',y='price',data=data)
sns.barplot(x='fuelType',y='price',data=data)
sns.scatterplot(x='mpg',y='price',data=data)
sns.scatterplot(x='engineSize',y='price',data=data)
sns.scatterplot(x='model',y='price',data=data)
data_x= data[['transmission','fuelType','tax','year','mileage','mpg']]
data_y= data['price']
data_x
data_y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_x, data_y, test_size=0.2, random_state=0)
x_train.shape, y_train.shape
from sklearn import svm
sv=svm.SVR()
sv.fit(x_train,y_train)
yp=sv.predict(x_test)
yp
sv.score(x_test,y_test)
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
tree.fit(x_train,y_train)
ypp=tree.predict(x_test)
ypp
tree.score(data_x,data_y)