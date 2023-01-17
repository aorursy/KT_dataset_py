# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#creating a dataframe out of the input provided using pandas library

df = pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head(2)
# Checking the data type of the columns
df.info()
df.isna().sum()
# Using seaborn- boxplot to visualize the relationship between dependent variable(SP) and other independent variables.
plt.title('Fuel_Type Vs Selling_Price')
sns.boxplot(df['Fuel_Type'], df['Selling_Price'])

plt.title('Transmission Vs Selling_Price')
sns.boxplot(df['Transmission'], df['Selling_Price'])

#we could see that automatic transmission cars are costlier than manual.
all_cls=df.columns
numeric_cols=df._get_numeric_data().columns.to_list()
cat_cols=list(set(all_cls)-set(numeric_cols))
df.drop(['Car_Name'],inplace=True,axis=1)
df['age']=df['Year']-2020
fig_dims = (18,6)
fig, ax = plt.subplots(figsize=fig_dims)
plt.title('Age Vs Selling_Price')
sns.boxplot(df['age'], df['Selling_Price'], ax=ax, data=df)
df.drop(['Year'],inplace=True,axis=1)
df=pd.get_dummies(df,drop_first=True)
y=df.pop('Selling_Price')
X=df
from sklearn.ensemble import ExtraTreesRegressor
md=ExtraTreesRegressor()
md.fit(X,y)

plt.figure(figsize=(18,8))
sns.barplot(x=df.columns, y=md.feature_importances_)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
y_pred
from sklearn.metrics import mean_squared_error
rmse_value=mean_squared_error(y_test, y_pred,squared=False)
print(rmse_value)