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
# Importing the libraries 
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()

# Initializing the dataframe
data = pd.DataFrame(boston.data)
# Adding feature name to the dataframe
data.columns = boston.feature_names
# the upper 5 data
data.head()
# the lower 5 data
data.tail()
data.shape
data.columns
data.dtypes
data.info()
data.isnull().sum()
data['CRIM'].unique()
data['ZN'].unique()
data['INDUS'].unique()
data['NOX'].unique()
data['RM'].unique()
data['AGE'].unique()
data['DIS'].unique()
data['RAD'].unique()
data['PTRATIO'].unique()
data['B'].unique()
data['LSTAT'].unique()
# Assign the target column
data['PRICE'] = boston.target 
# Shows the statistical summary
data.describe()
data.CRIM.quantile(0.999)
cor=data.corr()
#Heatmap for visualisation of correlation analysis
plt.figure(figsize=(10,8))
sns.heatmap(cor,annot=True,cmap='coolwarm')
#when we write annot= True , it shows the values .
plt.show()
x=data[['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']]
y=data['PRICE']
# split data into train and test
from sklearn.model_selection import train_test_split
xtr,xts,ytr,yts = train_test_split(x,y,test_size=0.2)
# we have to split the data into 80% as train and 20% as test so we have specified test_size as 0.2
print(x.shape)
print(xtr.shape)
print(xts.shape)
print(y.shape)
print(ytr.shape)
print(yts.shape)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtr, ytr)
y_pred = regressor.predict(xts)
#calculating r2score
from sklearn.metrics import r2_score
r2_score(yts,y_pred)
#To find the error
from sklearn.metrics import mean_squared_error
mean_squared_error(yts,y_pred)
# Visualizing the differences between actual prices and predicted values
plt.scatter(yts,y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg = RandomForestRegressor()

# Train the model using the training sets 
reg.fit(xtr, ytr)
y_pred = reg.predict(xts)
r2_score(yts,y_pred)
mean_squared_error(yts,y_pred)
# Visualizing the differences between actual prices and predicted values
plt.scatter(yts,y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
