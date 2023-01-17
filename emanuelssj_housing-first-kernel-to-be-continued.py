# Core

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Imputer, grid, robust, cross

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import Imputer

from sklearn.model_selection import GridSearchCV, cross_val_score

from scipy.stats import skew



# Machine Learning

import lightgbm as lgb

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from xgboost import XGBRegressor



import warnings

warnings.filterwarnings('ignore')



path_test = '../input/test.csv'



train_data = pd.read_csv('../input/train.csv' , index_col= 0)

test_data = pd.read_csv(path_test  , index_col= 0)



train_data = train_data[train_data.GrLivArea < 4500]



label = train_data[['SalePrice']]

train_data.drop('SalePrice' , axis = 1 , inplace=True)

train_data.head(3)
# Divide in categorical columns and numerical columns

numerical_col = []

cat_col = []

for x in train_data.columns:

    if train_data[x].dtype == 'object':

        cat_col.append(x)

        print(x+': ' + str(len(train_data[x].unique())))

    else:

        numerical_col.append(x)

        

print('CAT col \n', cat_col)

print('Numerical col\n')

print(numerical_col)
# Checking skew

numerical = train_data.select_dtypes(exclude='object').copy()



fig = plt.figure(figsize=(12,18))

for i in range(len(numerical.columns)):

    fig.add_subplot(9,4,i+1)

    sns.distplot(numerical.iloc[:,i].dropna())

    plt.xlabel(numerical.columns[i])



plt.tight_layout()

plt.show()
# Checking corr and choosing columns to be removed



transformed_corr = train_data.corr()

plt.figure(figsize=(12,10))

sns.heatmap(transformed_corr)



# Highly-correlated:

# GarageCars and GarageArea

# YearBuilt and GarageYrBlt

# GrLivArea_log1p and TotRmsAbvGrd



columns_highCorr_drop = ['GarageCars', 'GarageYrBlt', 'TotRmsAbvGrd', '2ndFlrSF', '1stFlrSF']



# Drop Highly_correlated

train_data = train_data.drop(columns_highCorr_drop, axis=1)

test_data = test_data.drop(columns_highCorr_drop, axis=1)
print(numerical_col)

numerical_col.remove('TotRmsAbvGrd')

numerical_col.remove('GarageYrBlt')

numerical_col.remove('GarageCars')

numerical_col.remove('2ndFlrSF')

numerical_col.remove('1stFlrSF')

print(cat_col)
# Skew data reduce

len_train = train_data.shape[0]



houses=pd.concat([train_data,test_data], sort=False)

skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skew_df=pd.DataFrame({'Skew':skew})

skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]



train_data=houses[:len_train]

test_data=houses[len_train:]



# Reduce skew in target - Log

label = np.log(label)
# Cleaning numerical columns from train_data and test_data

imputer = Imputer(missing_values='NaN' , strategy='mean' , axis = 0)

imputer = imputer.fit(train_data[numerical_col])

train_num = imputer.transform(train_data[numerical_col])



test_num = imputer.transform(test_data[numerical_col])

print(train_num.shape)

print(test_num.shape)
# Cleaning categorical columns from train_data and test_data



train_cat = train_data[cat_col]

test_cat = test_data[cat_col]



dropp = ['MiscFeature' , 'PoolQC' , 'Fence' ,'Alley' ]

train_cat.drop(columns=dropp , axis=1, inplace=True)

train_cat = train_cat.astype('category')

test_cat.drop(columns=dropp , axis=1, inplace=True)

test_cat = test_cat.astype('category')



# Fill null values with the most frequent attribute

most_freq = {}

for col in train_cat.columns:

    p = train_cat[col].mode()[0]

    train_cat[col].fillna(p, inplace=True)

    most_freq[col] = p



for col in train_cat.columns:

    test_cat[col].fillna(most_freq[col], inplace=True)
# Converting to dataframe



train_num = pd.DataFrame(train_num)

train_num.head(2)

test_num = pd.DataFrame(test_num)

test_num.head(2)
# Encode categoricals values

for col in train_cat:

    train_cat[col] = train_cat[col].cat.codes

for col in test_cat:

    test_cat[col] = test_cat[col].cat.codes

train_cat.head(2)
# Same index for numerical and categorical dataframes

train_num.index = train_cat.index

test_num.index = test_cat.index



# Get dummies for categorical columns

train_cat = pd.get_dummies(train_cat)

test_cat = pd.get_dummies(test_cat)



# Join the 2 datas into one

train_ = train_num.join(train_cat)

test_ = test_num.join(test_cat)



# Scaling the data

scalar = RobustScaler()

train_ = scalar.fit_transform(train_)

test_ = scalar.transform(test_)
import lightgbm as lgb

lightgbm = lgb.LGBMRegressor(objective='regression', 

                                       num_leaves=8,

                                       learning_rate=0.03, 

                                       n_estimators=4000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )

scores = cross_val_score(lightgbm, train_, label, cv=5).mean()

scores
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha =0.001, random_state=1)

scores = cross_val_score(lasso_model, train_, label, cv=5).mean()

scores
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.002, random_state=5)

scores = cross_val_score(ridge_model, train_, label, cv=5).mean()

scores
# # XGBRegressor - Hyperparameter Tuning



# xgbRegressor = XGBRegressor()

# n_estimators = range(70, 100)

# learning_rate = np.arange(0.4, 0.7, 0.1)



# ## Search grid for optimal parameters

# param_grid = {"n_estimators" : n_estimators}



# model_xgb = GridSearchCV(xgbRegressor, param_grid = param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs= 4, verbose = 1)



# model_xgb.fit(train_,label)



# # Best score

# print(model_xgb.best_score_)



# #best estimator

# model_xgb.best_estimator_
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=4000, learning_rate=0.05)

scores = cross_val_score(xgb_model, train_, label, cv=5).mean()

scores
# remove log from prediction

def inv_y(transformed_y):

    return np.exp(transformed_y)



# Choose model

lasso_model.fit(train_, label)

pre = lasso_model.predict(test_)
submission = pd.DataFrame({'Id': pd.read_csv(path_test).Id,

                       'SalePrice': inv_y(pre)})



submission.to_csv('submission.csv', index=False)