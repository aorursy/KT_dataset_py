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
# loading packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# loading data

data = pd.read_csv('/kaggle/input/datasets_33080_43333_car data.csv')

data.head()
data.info()
data.isnull().sum()
# taking object datatypes only

data.select_dtypes(include= np.object).head()
# creating a function for getting unique values from dataset

def get_unique_values(dataset):

    df = dataset.select_dtypes(include = np.object)

    cols = list(df.columns)

    for i in cols:

        print('{}: {}'.format(i,df[i].unique()), '\n')
get_unique_values(data)
data['Owner'].unique()
data['Year'].unique()
# dropping unnecessary attributes

data.drop(columns=['Car_Name'], axis = 1,inplace= True)

data.head()
# considering only numeric attributes

data.iloc[:, 1:4].describe().T
# data distribution using histograms

plt.style.use('seaborn')

data.iloc[:, 1:4].hist(figsize = (12,12))

plt.show()
## scatter plot for linearity check and spread on target variable/attribute

for i in list(data.columns)[2:4]:

    sns.relplot(x = i, y = 'Selling_Price', data= data)

    plt.show()
## plotting by category



for i in list(data.columns)[2:4]:

    for j in list(data.select_dtypes(include= np.object).columns):

        sns.relplot(x = i, y = 'Selling_Price',hue = j,  data= data)

    plt.show()
## correlation

data.iloc[:, [0,2,3]].corr()
# to check yearly change

for i in list(data.columns)[1:4]:

    sns.lineplot(x = 'Year', y = i, data = data, color="coral")

    plt.show()
# to check yearly change with category wise

for i in list(data.columns)[1:4]:

    sns.lineplot(x = 'Year', y = i, data = data, hue = 'Fuel_Type' ,color="coral")

    plt.show()
# to check yearly change with category wise

for i in list(data.columns)[1:4]:

    sns.lineplot(x = 'Year', y = i, data = data, hue = 'Seller_Type' ,color="coral")

    plt.show()
# to check yearly change with category wise

for i in list(data.columns)[1:4]:

    sns.lineplot(x = 'Year', y = i, data = data, hue = 'Transmission' ,color="coral")

    plt.show()
sns.barplot(x = 'Fuel_Type', y = 'Selling_Price', data = data, order = ['Diesel', 'Petrol', 'CNG'])

plt.show()
data.columns
list(data.columns)[4:len(list(data.columns))]
# selling price on categorical data

for i in list(data.columns)[4:len(list(data.columns))]:

    sns.barplot(x = i, y = 'Selling_Price', data= data, )

    plt.show()
# change categorical values into numeric by getting dummies

data = pd.get_dummies(data,columns= ['Fuel_Type','Seller_Type', 'Transmission'], drop_first= True)

data.head()
X_data = data.drop(columns= 'Selling_Price', axis = 1)

y_data = data['Selling_Price']
# here I m using extra tree classifier but you can use decision tree or randomforest too

from sklearn.ensemble import ExtraTreesRegressor

et = ExtraTreesRegressor()

et.fit(X_data, y_data)
# check feature importance values

et.feature_importances_
#plot graph of feature importances for better visualization

feat_importances = pd.Series(et.feature_importances_, index=X_data.columns)

feat_importances.nlargest(5).plot(kind='barh')

plt.show()
# converting into X and y arrays

X = X_data.iloc[:].values

y = y_data.iloc[:].values
# splitting the data into train and test

# random state is any orbitary number and it will help you to get same reslut as am I when you run this notebook.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 10)
# scaling the data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# models that I want to perform 



from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
# linear regression

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)
# training scores and testing scores

print("training score :{}".format(lr_model.score(X_train, y_train)))

print("testing score :{}".format(lr_model.score(X_test, y_test)))
# predictions

y_pred = lr_model.predict(X_test)

y_pred
# error metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('mae score : {}'.format(mean_absolute_error(y_test, y_pred)))

print('mse score : {}'.format(mean_squared_error(y_test, y_pred)))

print('rmse score : {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

print('R2 score : {}'.format(r2_score(y_test, y_pred)))
# decision tree

dt_model = DecisionTreeRegressor()

dt_model.fit(X_train, y_train)
print("training score :{}".format(dt_model.score(X_train, y_train)))

print("testing score :{}".format(dt_model.score(X_test, y_test)))
y_pred = dt_model.predict(X_test)

y_pred
print('mae score : {}'.format(mean_absolute_error(y_test, y_pred)))

print('mse score : {}'.format(mean_squared_error(y_test, y_pred)))

print('rmse score : {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

print('R2 score : {}'.format(r2_score(y_test, y_pred)))
# Randomforest model

rf_model = RandomForestRegressor()

rf_model.fit(X_train, y_train)
print("training score :{}".format(rf_model.score(X_train, y_train)))

print("testing score :{}".format(rf_model.score(X_test, y_test)))
y_pred = rf_model.predict(X_test)

y_pred
print('mae score : {}'.format(mean_absolute_error(y_test, y_pred)))

print('mse score : {}'.format(mean_squared_error(y_test, y_pred)))

print('rmse score : {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

print('R2 score : {}'.format(r2_score(y_test, y_pred)))
# xgboost

import xgboost as xgb

xg = xgb.XGBRegressor()

xg.fit(X_train, y_train)
print("training score :{}".format(xg.score(X_train, y_train)))

print("testing score :{}".format(xg.score(X_test, y_test)))
y_pred = xg.predict(X_test)

y_pred
print('mae score : {}'.format(mean_absolute_error(y_test, y_pred)))

print('mse score : {}'.format(mean_squared_error(y_test, y_pred)))

print('rmse score : {}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

print('R2 score : {}'.format(r2_score(y_test, y_pred)))