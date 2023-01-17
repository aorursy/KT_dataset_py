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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
# Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import median_absolute_error
from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv('../input/used-car-dataset-ford-and-mercedes/audi.csv')
data.head(10)
col=data.columns.values
print(col)
data.info()
data[ 'fuelType'].unique()
data[ 'model'].unique()
data['transmission'].unique()
data.describe()
sns.distplot(data['price'])
data = pd.concat([data['price'], data['mileage']], axis=1)
data.plot.scatter(x='mileage', y='price', ylim=(0,100000));
 
data = pd.concat([data['price'], data['year']], axis=1)
data.plot.scatter(x='year', y='price', ylim=(0,900000));
data = pd.concat([data['price'], data['engineSize']], axis=1)
data.plot.scatter(x='engineSize', y='price', ylim=(0,100000));
#scatterplot
sns.set()
cols = ['model', 'year', 'price', 'transmission', 'mileage', 'fuelType',
       'tax', 'mpg', 'engineSize']
sns.pairplot(data[cols], size = 2.5)
plt.show();
#correlation matrix
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#sns.heatmap(corrmat,fmt=".2f",cmap='coolwarm', annot=True)
le = LabelEncoder()

data["le_transmission"] = le.fit_transform(data["transmission"])
data["le_fuelType"] = le.fit_transform(data["fuelType"])
data["le_model"] = le.fit_transform(data["model"])
data.head()
data_1=data.drop(columns = ["transmission", "fuelType",'model'])
data_1.head(10)
#Standard Scaler for Data

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
data_1 = scaler.fit_transform(data_1)
 
 
X=data_1.drop(columns=['price','mpg'])
y=data_1['price']
X.shape
y.shape

#Splitting data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)

#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)
#Applying Lasso Regression Model 

'''
sklearn.linear_model.Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=
                           False, copy_X=True, max_iter=1000, tol=0.0001,
                           warm_start=False, positive=False, random_state=None,selection='cyclic')
'''

LassoRegressionModel = Lasso(alpha=.10,random_state=33,max_iter=100000,normalize=False)
LassoRegressionModel.fit(X_train, y_train)

#Calculating Details
print('Lasso Regression Train Score is : ' , LassoRegressionModel.score(X_train, y_train))
print('Lasso Regression Test Score is : ' , LassoRegressionModel.score(X_test, y_test))
print('Lasso Regression Coef is : ' , LassoRegressionModel.coef_)
print('Lasso Regression intercept is : ' , LassoRegressionModel.intercept_)
#Calculating Prediction
y_pred = LassoRegressionModel.predict(X_test)
print('Predicted Value for Lasso Regression is : ' , y_pred[:10])
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Absolute Error Value is : ', MAEValue)

#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')  
print('Mean Squared Error Value is : ', MSEValue)
#Calculating Median Absolute Error
MdSEValue = median_absolute_error(y_test, y_pred)
print('Median Absolute Error Value is : ', MdSEValue )
#Applying Random Forest Regressor Model 

'''
sklearn.ensemble.RandomForestRegressor(n_estimators='warn', criterion=’mse’, max_depth=None,
                                       min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                                       max_features=’auto’, max_leaf_nodes=None,min_impurity_decrease=0.0,
                                       min_impurity_split=None, bootstrap=True,oob_score=False, n_jobs=None,
                                       random_state=None, verbose=0,warm_start=False)
'''

RandomForestRegressorModel = RandomForestRegressor(n_estimators=500,max_depth=20, random_state=33)
RandomForestRegressorModel.fit(X_train, y_train)
#Calculating Details
print('Random Forest Regressor Train Score is : ' , RandomForestRegressorModel.score(X_train, y_train))
print('Random Forest Regressor Test Score is : ' , RandomForestRegressorModel.score(X_test, y_test))
print('Random Forest Regressor No. of features are : ' , RandomForestRegressorModel.n_features_)
#Calculating Prediction
y_pred = RandomForestRegressorModel.predict(X_test)
print('Predicted Value for Random Forest Regressor is : ' , y_pred[:10])
#Calculating Mean Absolute Error
MAEValue = mean_absolute_error(y_test, y_pred, multioutput='uniform_average') 
print('Mean Absolute Error Value is : ', MAEValue)
#Calculating Mean Squared Error
MSEValue = mean_squared_error(y_test, y_pred, multioutput='uniform_average')  
print('Mean Squared Error Value is : ', MSEValue)