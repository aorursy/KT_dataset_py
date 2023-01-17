# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory # ref https://www.kaggle.com/vikrishnan/house-sales-price-using-regression

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data_df=pd.read_csv('/kaggle/input/boston-house-prices/housing.csv', header=None, delim_whitespace=True, names=column_names) #, delimiter=r"\s+")
data_df.head()
print(data_df.shape)
data_df.info()
data_df.isnull().sum()
data_df.duplicated().any()
data_df.describe()
data_df.corr(method='pearson')
data_df.hist(bins=12,figsize=(12,10),grid=False);
y=data_df['MEDV']

X=data_df.drop('MEDV',axis = 1)
X.head()
y.head()
from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



sns.distplot(y, hist=True);

fig = plt.figure()

res = stats.probplot(y, plot=plt)
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_squared_error



results = []



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)





pipeLR=Pipeline([('scaler', StandardScaler()), ('LR', LinearRegression())])



# the benefit of using K-Fold is that we could calculate the cross validation value using some of the methods of scoring 

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

cv_results = cross_val_score(pipeLR, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')



pipeLR.fit(X_train, y_train)



# this is the scaled LR

print("Score for scaledLR: ", pipeLR.score(X_test, y_test))

# the mean result (10 data) of negative mean squared error

print("Score for scaledLR using cross_val_score: ", cv_results.mean())



import sklearn

sklearn.metrics.SCORERS.keys()
pipeLASSO=Pipeline([('scaler', StandardScaler()), ('LASSO', Lasso())])



# the benefit of using K-Fold is that we could calculate the cross validation value using some of the methods of scoring 

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

cv_results = cross_val_score(pipeLASSO, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')



pipeLASSO.fit(X_train, y_train)



# this is the scaled LR

print("Score for scaledLASSO: ", pipeLASSO.score(X_test, y_test))

# the mean result (10 data) of negative mean squared error

print("Score for scaledLASSO using cross_val_score: ", cv_results.mean())
pipeEN=Pipeline([('scaler', StandardScaler()), ('ElasticNet', ElasticNet())])



# the benefit of using K-Fold is that we could calculate the cross validation value using some of the methods of scoring 

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

cv_results = cross_val_score(pipeEN, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')



pipeEN.fit(X_train, y_train)



# this is the scaled LR

print("Score for pipeEN: ", pipeEN.score(X_test, y_test))

# the mean result (10 data) of negative mean squared error

print("Score for pipeEN using cross_val_score: ", cv_results.mean())
pipeKNNReg=Pipeline([('scaler', StandardScaler()), ('KNeighborsRegressor', KNeighborsRegressor())])



# the benefit of using K-Fold is that we could calculate the cross validation value using some of the methods of scoring 

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

cv_results = cross_val_score(pipeKNNReg, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')



pipeKNNReg.fit(X_train, y_train)



# this is the scaled LR

print("Score for pipeKNNReg: ", pipeKNNReg.score(X_test, y_test))

# the mean result (10 data) of negative mean squared error

print("Score for pipeKNNReg using cross_val_score: ", cv_results.mean())
pipeGBM=Pipeline([('scaler', StandardScaler()), ('GradientBoostingRegressor', GradientBoostingRegressor())])



# the benefit of using K-Fold is that we could calculate the cross validation value using some of the methods of scoring 

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

cv_results = cross_val_score(pipeGBM, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')



pipeGBM.fit(X_train, y_train)



# this is the scaled LR

print("Score for pipeGBM: ", pipeGBM.score(X_test, y_test))

# the mean result (10 data) of negative mean squared error

print("Score for pipeGBM using cross_val_score: ", cv_results.mean())
pipeGBM=Pipeline([('scaler', StandardScaler()), ('GradientBoostingRegressor', GradientBoostingRegressor(random_state=0, n_estimators=400))])



# the benefit of using K-Fold is that we could calculate the cross validation value using some of the methods of scoring 

kfold = KFold(n_splits=10, shuffle=True, random_state=0)

cv_results = cross_val_score(pipeGBM, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')



pipeGBM.fit(X_train, y_train)



#scaler = StandardScaler().fit(X_train)



#X_test_scaled=scaler.transform(X_test)



predictions = pipeGBM.predict(X_test)
print(mean_squared_error(y_test, predictions))
pred_df=pd.DataFrame({"Original Price of House": y_test, "Prediction Price of House": predictions})
pred_df.head(10)