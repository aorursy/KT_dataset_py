import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

import os



import re



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale

from sklearn.feature_selection import RFE

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.pipeline import make_pipeline

from sklearn.metrics import r2_score



# hide warnings

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
test.head()
train.shape
test.shape
train.isnull().sum()
test.isnull().sum()
round(100*(train.isnull().sum())/len(train.index))
round(100*(test.isnull().sum())/len(test.index))
train.describe(include="all")
test.describe(include="all")
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#Deleting outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



#Check the graphic again

fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
#concatenate the train and test data in the same dataframe



ntrain = train.shape[0]

ntest = test.shape[0]

y = train.SalePrice.values

surprise = pd.concat((train, test)).reset_index(drop=True)

surprise.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(surprise.shape))
surprise_na = (surprise.isnull().sum() / len(surprise)) * 100

surprise_na = surprise_na.drop(surprise_na[surprise_na == 0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({'Missing Ratio' :surprise_na})

missing_data.head(20)
f, ax = plt.subplots(figsize=(10, 5))

plt.xticks(rotation='90')

sns.barplot(x=surprise_na.index, y=surprise_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
#Correlation map to see how features are correlated with SalePrice

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)
for col in ('PoolQC','MiscFeature','Alley','Fence','FireplaceQu','MSSubClass','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual', 'BsmtCond','MasVnrType','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    surprise[col] = surprise[col].fillna('None')
for col in ('MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    surprise[col] = surprise[col].fillna(0)
surprise['MSZoning'] = surprise['MSZoning'].fillna(surprise['MSZoning'].mode()[0])

surprise['Electrical'] = surprise['Electrical'].fillna(surprise['Electrical'].mode()[0])

surprise['KitchenQual'] = surprise['KitchenQual'].fillna(surprise['KitchenQual'].mode()[0])

surprise['Exterior1st'] = surprise['Exterior1st'].fillna(surprise['Exterior1st'].mode()[0])

surprise['Exterior2nd'] = surprise['Exterior2nd'].fillna(surprise['Exterior2nd'].mode()[0])

surprise['SaleType'] = surprise['SaleType'].fillna(surprise['SaleType'].mode()[0])
surprise = surprise.drop(['Utilities'], axis=1)
#Functional : data description says NA means typical



surprise["Functional"] = surprise["Functional"].fillna("Typ")


surprise["LotFrontage"] = surprise.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
surprise.isnull().values.any()
#Transforming some numerical variables that are really categorical

surprise['MSSubClass'] = surprise['MSSubClass'].apply(str)

surprise['OverallCond'] = surprise['OverallCond'].astype(str)

surprise['YrSold'] = surprise['YrSold'].astype(str)

surprise['MoSold'] = surprise['MoSold'].astype(str)



surprise.shape
surprise = pd.get_dummies(surprise)

print(surprise.shape)
train = surprise[:ntrain]

test = surprise[ntrain:]
train.shape
test.shape
# scaling the features

from sklearn.preprocessing import scale



# storing column names in cols, since column names are (annoyingly) lost after 

# scaling (the df is converted to a numpy array)

cols = train.columns

train = pd.DataFrame(scale(train))

train.columns = cols

train.columns
# split into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, y, 

                                                    train_size=0.7,

                                                    test_size = 0.3, random_state=100)
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500,1000 ]}





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

cv_results = cv_results[cv_results['param_alpha']<=200]

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
model_cv.best_params_
alpha = 500

ridge = Ridge(alpha=alpha)



ridge.fit(X_train, y_train)

ridge.coef_ 
# lasso regression

lm = Lasso(alpha=1000)

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

cols = train.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))
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

cv_results.head()
# plot

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('r2 score')

plt.xscale('log')

plt.show()
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
model_cv.best_params_
alpha = 1000



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train, y_train)
lasso.coef_
# lasso model parameters

model_parameters = list(lm.coef_)

model_parameters.insert(0, lm.intercept_)

model_parameters = [round(x, 3) for x in model_parameters]

cols = train.columns

cols = cols.insert(0, "constant")

list(zip(cols, model_parameters))
# lasso regression

lm1 = Lasso(alpha=1000)

lm1.fit(X_train, y_train)



# predict

y_train_pred = lm1.predict(X_train)

print(r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm1.predict(X_test)

print(r2_score(y_true=y_test, y_pred=y_test_pred))
# Ridge regression

lm2 = Ridge(alpha=500)

lm2.fit(X_train, y_train)



# predict

y_train_pred = lm2.predict(X_train)

print(r2_score(y_true=y_train, y_pred=y_train_pred))

y_test_pred = lm2.predict(X_test)

print(r2_score(y_true=y_test, y_pred=y_test_pred))
preds = lm1.predict(test)

sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice'] = preds

sub.to_csv('house_sub.csv',index=False)