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
# hide warnings

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

import os       
from sklearn.metrics import r2_score
# reading the dataset

housing = pd.read_csv("../input/train.csv")
housing.head()
housing.shape
housing.columns
# summary of the dataset: 1460 rows, 81 columns, no null values

print(housing.info())
housing.describe()
# Checking the percentage of missing values

round(100*(housing.isnull().sum()/len(housing.index)), 2)
# Write your code for column-wise null count here

housing.isnull().sum()
#Replacing NA values per data dictionary

housing.PoolQC.replace(np.NaN, 'No Pool', inplace=True)

housing.Alley.replace(np.NaN, 'No alley access', inplace=True)

housing.BsmtQual.replace(np.NaN, 'No Basement', inplace=True)

housing.BsmtCond.replace(np.NaN, 'No Basement', inplace=True)

housing.BsmtExposure.replace(np.NaN, 'No Basement', inplace=True)

housing.BsmtFinType1.replace(np.NaN, 'No Basement', inplace=True)

housing.BsmtFinType2.replace(np.NaN, 'No Basement', inplace=True)

housing.FireplaceQu.replace(np.NaN, 'No Fireplace', inplace=True)

housing.GarageType.replace(np.NaN, 'No Garage', inplace=True)

housing.GarageFinish.replace(np.NaN, 'No Garage', inplace=True)

housing.GarageFinish.replace(np.NaN, 'No Garage', inplace=True)

housing.GarageQual.replace(np.NaN, 'No Garage', inplace=True)

housing.GarageCond.replace(np.NaN, 'No Garage', inplace=True)

housing.Fence.replace(np.NaN, 'No Fence', inplace=True)

housing.MiscFeature.replace(np.NaN, 'None', inplace=True)

housing.MasVnrType.replace(np.NaN, 'None', inplace=True)
# Checking the percentage of missing values

round(100*(housing.isnull().sum()/len(housing.index)), 2)
#checking if any duplicate values in the df

print(any(housing.duplicated()))  
housing.LotFrontage.describe(percentiles=[.25,.5,.75,.90,.95,.99])
housing.LotFrontage.replace(np.NaN, 70.049958, inplace=True)
housing = housing.drop('GarageYrBlt', axis=1)
# Checking the percentage of missing values

round(100*(housing.isnull().sum()/len(housing.index) ), 2)
# Write your code for dropping the rows here

housing[housing.columns].isnull().sum().value_counts()
housing.columns[housing.isna().any()].tolist()
# Write your code for dropping the rows here

housing[housing.columns].isnull().sum().value_counts()

housing = housing.dropna(axis=0, subset=['MasVnrArea'])
housing = housing.dropna(axis=0, subset=['Electrical'])
housing.info()
# all numeric (float and int) variables in the dataset

housing_numeric = housing.select_dtypes(include=['float64', 'int64'])

housing_numeric.head()
housing_numeric = housing_numeric.drop(['MSSubClass'], axis=1)
# correlation matrix

cor = housing_numeric.corr()

cor
# plotting correlations on a heatmap



# figure size

plt.figure(figsize=(25,18))



# heatmap

sns.heatmap(cor, cmap="YlGnBu", annot=True)

plt.show()

#histogram

sns.distplot(housing['SalePrice']);
#scatter plot grlivarea/saleprice

data = pd.concat([housing['SalePrice'], housing['GrLivArea']], axis=1)

data.plot.scatter(x="GrLivArea", y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

data = pd.concat([housing['SalePrice'], housing['TotalBsmtSF']], axis=1)

data.plot.scatter(x="TotalBsmtSF", y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

data = pd.concat([housing['SalePrice'], housing['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x="OverallQual", y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
data = pd.concat([housing['SalePrice'], housing['YearBuilt']], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x="YearBuilt", y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = housing.corr().nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(housing[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# converting MSSubClass to categorical

housing['MSSubClass'] = housing['MSSubClass'].astype('object')

housing.info()



# creating dummy variables for categorical variables

# subset all categorical variables

housing_categorical = housing.select_dtypes(include=['object'])

housing_categorical.head()

# convert into dummies

housing_dummies = pd.get_dummies(housing_categorical, drop_first=True)

housing_dummies.head()
 # drop categorical variables 

housing_final = housing.drop(list(housing_categorical.columns), axis=1)
# concat dummy variables with X

housing_final = pd.concat([housing_final, housing_dummies], axis=1)
housing_final.shape
housing_final
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

#housing_final = pd.DataFrame(scaler.fit_transform(housing_final), columns=housing_final.columns)

#housing_final
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = housing_final.drop(['SalePrice','Id'], axis=1)



X.head()
# scaling the features

from sklearn.preprocessing import scale



# storing column names in cols, since column names are (annoyingly) lost after 

# scaling (the df is converted to a numpy array)

cols = X.columns

X = pd.DataFrame(scale(X))

X.columns = cols

X.columns
# Putting response variable to y

y = housing_final['SalePrice']

y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=200)
# linear regression

lm = LinearRegression()

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

r2_score(y_true=y_train, y_pred=y_train_pred)
y_test_pred = lm.predict(X_test)



r2_score(y_true=y_test, y_pred=y_test_pred)
# model coefficients

cols = X.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))
# lasso regression

lm = Lasso(alpha=0.001)

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

print(r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm.predict(X_test)

print(r2_score(y_true=y_test, y_pred=y_test_pred))
# lasso model parameters

model_parameters = list(lm.coef_)

model_parameters.insert(0, lm.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = X.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))

# grid search CV



# set up cross validation scheme

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# specify range of hyperparameters

params = {'alpha': [0.001, 0.01, 1.0, 5.0, 10.0]}



# grid search

# lasso model

model = Lasso()

model_cv = GridSearchCV(estimator = model, param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        return_train_score=True, verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plot

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('r2 score')

plt.xscale('log')

plt.show()
# model with optimal alpha

# lasso regression

lm = Lasso(alpha=500)

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

print(r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm.predict(X_test)

print(r2_score(y_true=y_test, y_pred=y_test_pred))
# lasso model parameters

model_parameters = list(lm.coef_)

model_parameters.insert(0, lm.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = X.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))
# set up cross validation scheme

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
# specify range of hyperparameters

params = {'alpha': [0.001, 0.01, 1.0, 5.0, 10.0,50.0,100.0,200.0,300.0,350.0,400.0,450.0,500.0,550.0,600.0,650.0,700.0,750.0,800.0,850.0]}

# grid search

# lasso model

model = Lasso()

model_cv = GridSearchCV(estimator = model, param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        return_train_score=True, verbose = 1)            

model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plot

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('r2 score')

plt.xscale('log')

plt.show()
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}



folds = 5



lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha =100



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train) 
lasso.coef_
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.5, 1.0, 2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0 ,10.0, 11.0,12.0,13.0,14.0, 15.0,16.0,17.0,18.0,19.0, 20.0,25.0, 30.0, 35.0, 40.0,45.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0 ]}





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

cv_results = cv_results[cv_results['param_alpha']<=100]

cv_results.head()
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
# model with optimal alpha

# Ridge regression

lm = Ridge(alpha=90.0)

lm.fit(X_train, y_train)



# predict

y_train_pred = lm.predict(X_train)

print(r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm.predict(X_test)

print(r2_score(y_true=y_test, y_pred=y_test_pred))