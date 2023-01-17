# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import all libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import re



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline





import warnings # supress warnings

warnings.filterwarnings('ignore')
# import train.csv

df = pd.read_csv("../input/train.csv")
#display the first ten columns of the dataset to get a feel of the dataset.



df.head(10)
# summary of the dataset: 1460 rows, 81 columns, some have null values

print(df.info())
df.describe()
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(25,15))



# heatmap



corr= df.corr()

sns.heatmap(corr, cmap="YlGnBu", annot=True)

plt.show()
# paiwise scatter plot



plt.figure(figsize=(20, 10))

sns.pairplot(df)

plt.show()
# scatter plot

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df[cols])

plt.show()
# dependent variable is  salesprice so separating dependent and independent variable

dependent_variable = df['SalePrice']

dependent_variable.head()
# check the distribution of target variable

sns.distplot(dependent_variable,hist=True)
# visualise area-price relationship

sns.regplot(x="OverallQual", y="SalePrice", data=df, fit_reg=False)
sns.regplot(x="GrLivArea", y="SalePrice", data=df, fit_reg=False)
sns.regplot(x="GarageArea", y="SalePrice", data=df, fit_reg=False)
sns.regplot(x="TotalBsmtSF", y="SalePrice", data=df, fit_reg=False)
plt.subplots(figsize=(16,10))

plt.xticks(rotation='90')

sns.boxplot(x=df['YearBuilt'], y=df['SalePrice'])
df= df.fillna("NO")
# all numeric (float and int) variables in the dataset

df_numeric = df.select_dtypes(include=['float64', 'int64'])

df_numeric.head()
# dropping MSSSubClass and ID 

df_numeric = df_numeric.drop(['MSSubClass','Id','OverallQual','OverallCond'], axis=1)

df_numeric.head()
# variable formats

df.info()
# converting those variables categorical

df['MSSubClass'] = df['MSSubClass'].astype('object')

df['OverallQual'] = df['OverallQual'].astype('object')

df['OverallCond'] = df['OverallCond'].astype('object')

df.info()
df = df.drop('Id', axis=1)
df.isnull().sum().sort_values(ascending = False).head(100)

# checking NA

# there are no missing values in the dataset

df.isnull().values.any()

#df.columns
# split into X and y

X= df.iloc[ : , :-1]



y = df['SalePrice']
# creating dummy variables for categorical variables



# subset all categorical variables

df_categorical = X.select_dtypes(include=['object'])

df_categorical.head()
# convert into dummies

df_dummies = pd.get_dummies(df_categorical, drop_first=True)

df_dummies.head()
# drop categorical variables 

X = X.drop(list(df_categorical.columns), axis=1)
# concat dummy variables with X

X = pd.concat([X, df_dummies], axis=1)
# scaling the features

from sklearn.preprocessing import scale



# storing column names in cols, since column names are (annoyingly) lost after 

# scaling (the df is converted to a numpy array)

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols

X.columns
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=100)
from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import os







# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000,2000,5000,8000]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=8000]

cv_results
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha = 500

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_
# model with optimal alpha

# lasso regression

lm = Ridge(alpha=500)

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm.predict(X_test)

print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))