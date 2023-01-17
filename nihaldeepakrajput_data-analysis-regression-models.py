# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#load the second version that is v2 csv file
dataset = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv")
#check the contents of first 5 rows
dataset.head()
#check the length or number of rows
len(dataset)
#check the datatyoes
dataset.dtypes
#check for null values
dataset.isnull().sum()
#describe the dataset as a table
dataset.describe().T
#visualizing the number of rooms
sns.catplot("rooms", data = dataset, kind = "count", height = 6)
#visualizing animals
sns.catplot("animal", data = dataset, kind = "count", height = 6)
#visualizing furniture
sns.catplot("furniture", data = dataset, kind = "count", height = 6)
#renaming the total columns 
dataset = dataset.rename(columns = {'total (R$)' : 'Total'}, errors = 'raise')
#determine the price with respect to area 
plt.figure(figsize=(10,6))
sns.distplot(dataset[dataset.city=='São Paulo'].Total ,color='maroon',hist=False,label='São Paulo')
sns.distplot(dataset[dataset.city=='Porto Alegre'].Total ,color='black',hist=False,label='Porto Alegre')
sns.distplot(dataset[dataset.city=='Rio de Janeiro'].Total ,color='green',hist=False,label='Rio de Janeiro')
sns.distplot(dataset[dataset.city=='Belo Horizonte'].Total ,color='blue',hist=False,label='Belo Horizonte')
sns.distplot(dataset[dataset.city=='Campinas'].Total ,color='orange',hist=False,label='Campinas')
plt.xlim(0,20000)
#total with respect to animals
plt.figure(figsize=(10,6))
sns.distplot(dataset[dataset.animal=='acept'].Total ,color='maroon',hist=False,label='accept')
sns.distplot(dataset[dataset.animal=='not acept'].Total ,color='black',hist=False,label='not accept')
plt.xlim(0,20000)
#total with respect to furniture
plt.figure(figsize=(10,6))
sns.distplot(dataset[dataset.furniture=='furnished'].Total ,color='maroon',hist=False,label='Furnished')
sns.distplot(dataset[dataset.furniture=='not furnished'].Total ,color='black',hist=False,label='Unfurnished')
plt.xlim(0,20000)
#Label Encode the objects
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset.city = le.fit_transform(dataset.city)
dataset.floor = le.fit_transform(dataset.floor)
dataset.animal = le.fit_transform(dataset.animal)
dataset.furniture = le.fit_transform(dataset.furniture)
#check data type
dataset.dtypes
#check correlation
dataset.corr().style.background_gradient(cmap = 'coolwarm')
#dividing into X and y
X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values
#confirming with backward elimination 
import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
#setting significance level to 0.05
#Optimal values of X taken
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7]]
#Obtain the features having most importance
X_Modeled = backwardElimination(X_opt, SL)
X_Modeled
#load the values into X and y
#all features except the (R$) features are being taken
X = dataset.iloc[:,0:8].values
y = dataset.iloc[:,12:13].values
#split into training and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
#multivariate linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predict value
y_pred = regressor.predict(X_test)
#calculating the r2 score and mse
from sklearn.metrics import r2_score, mean_squared_error
#r2 score
r2_score(y_test, y_pred)
#mse value
mean_squared_error(y_test, y_pred)
#loading X with updated features
X = dataset.iloc[:,[3,5,6,7]].values
#splitting into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
#multivariate linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predict value
y_pred = regressor.predict(X_test)
#r2 score
r2_score(y_test, y_pred)
#mse value
mean_squared_error(y_test, y_pred)
#polynomial distribution
X = dataset.iloc[:,[3,5,6,7]].values
y = dataset.iloc[:,12:13].values
#fitting into polynomial equation
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(X)
#split into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly,y,test_size = 0.2, random_state = 0)
#multivariate linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predict value
y_pred = regressor.predict(X_test)
#r2 score
r2_score(y_test, y_pred)
#mse value
mean_squared_error(y_test, y_pred)
#load X and y
X = dataset.iloc[:,[3,5,6,7]].values
y = dataset.iloc[:,-1].values
#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_poly,y,test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5)
regressor.fit(X_train, y_train)
#predicting the value
y_pred = regressor.predict(X_test)
#r2 score
r2_score(y_test, y_pred)