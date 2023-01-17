# Let's get started!



# import libraries

# Essentials

import numpy as np

import pandas as pd

import datetime

import random



# Plots

import seaborn as sns

import matplotlib.pyplot as plt



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.ensemble import VotingRegressor



# Misc

from sklearn import model_selection

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error, r2_score

import sklearn.metrics as metrics

from scipy.stats import norm



df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')





df_train.info()
df_train.head(5)
train_ID = df_train['Id']

test_ID = df_test['Id']

df_train.drop(['Id'], axis=1, inplace=True)

df_test.drop(['Id'], axis=1, inplace=True)

df_train.shape, df_test.shape


df_train.describe()
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();

#Outlier analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#Outlier analysis saleprice/TotalBsmtSF

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#analysing 'SalePrice'



#descriptive statistics summary

df_train['SalePrice'].describe()

#histogram

sns.distplot(df_train['SalePrice']);

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column SalePrice

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
#Check the new distribution 

sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# Split features and labels

train_labels = df_train['SalePrice'].reset_index(drop=True)

train_features = df_train.drop(['SalePrice'], axis=1)

test_features = df_test
# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset

all_features = pd.concat([train_features, test_features]).reset_index(drop=True)

all_features.shape

# determine the percentage of missing values

def percent_missing(df):

    data = pd.DataFrame(df)

    df_cols = list(pd.DataFrame(data))

    dict_x = {}

    for i in range(0, len(df_cols)):

        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})

    

    return dict_x



missing = percent_missing(all_features)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percent of missing data')

df_miss[0:10]
# Visualize missing values

sns.set_style("white")

f, ax = plt.subplots(figsize=(8, 7))

sns.set_color_codes(palette='deep')

missing = round(df_train.isnull().mean()*100,2)

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar(color="b")

# Tweak the visual presentation

ax.xaxis.grid(False)

ax.set(ylabel="Percent of missing values")

ax.set(xlabel="Features")

ax.set(title="Percent missing data by feature")

sns.despine(trim=True, left=True)
# Some of the non-numeric predictors are stored as numbers; convert them into strings 

all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)

all_features['YrSold'] = all_features['YrSold'].astype(str)

all_features['MoSold'] = all_features['MoSold'].astype(str)

def handle_missing(features):

    # the data description states that NA refers to typical ('Typ') values

    print(features['Functional'].value_counts())

    features['Functional'] = features['Functional'].fillna('Typ')

    # Replace the missing values in each of the columns below with their mode

    #print(features['Electrical'].value_counts())

    features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])    

    #print(features['KitchenQual'].value_counts())

    features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])    

    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    

    # the data description stats that NA refers to "No Pool"

    features["PoolQC"] = features["PoolQC"].fillna("None")

    # Replacing the missing values with 0, since no garage = no cars in garage

    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

        features[col] = features[col].fillna(0)

    # Replacing the missing values with None

    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

        features[col] = features[col].fillna('None')

    # NaN values for these categorical basement features, means there's no basement

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

        features[col] = features[col].fillna('None')

        

    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood

    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



    # We have no particular intuition around how to fill in the rest of the categorical features

    # So we replace their missing values with None

    objects = []

    for i in features.columns:

        if features[i].dtype == object:

            objects.append(i)

    features.update(features[objects].fillna('None'))

        

    # And we do the same thing for numerical features, but this time with 0s

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    numeric = []

    for i in features.columns:

        if features[i].dtype in numeric_dtypes:

            numeric.append(i)

    features.update(features[numeric].fillna(0))    

    return features





all_features = handle_missing(all_features)
# Let's make sure we handled all the missing values

missing = percent_missing(all_features)

df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)

print('Percent of missing data')

df_miss[0:10]

all_features.head()
# Encode categorical features

# one hot encoding with dummies



all_features = pd.get_dummies(all_features).reset_index(drop=True)

all_features.shape
#Recreate training and test sets



X = all_features.iloc[:len(train_labels), :]

X_test = all_features.iloc[len(train_labels):, :]

X.shape, train_labels.shape, X_test.shape



#scale data before regression

scaler = MinMaxScaler()

scaler.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)



kf = KFold(n_splits=12, random_state=42, shuffle=True)
# model training

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, 

                       bagging_fraction=0.8,

                       bagging_freq=4, 

                       bagging_seed=8,

                       feature_fraction=0.2,

                       feature_fraction_seed=8,

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1,

                       random_state=42)



# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42)



# Ridge Regressor

ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))



# Support Vector Regressor

svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))



# Gradient Boosting Regressor

gbr = GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)  



# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)



votingreg = VotingRegressor([('lightgbm', lightgbm), ('xgboost', xgboost),('ridge',ridge),('svr',svr),('gbr',gbr),('rf',rf)])
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, train_labels, train_size = 0.8)



test_pred = votingreg.fit(x_train, y_train).predict(x_test)
print('MAE:', metrics.mean_absolute_error(y_test, test_pred))

print('MSE:', metrics.mean_squared_error(y_test, test_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))

print('r2_score:',r2_score(y_test, test_pred))
# prediction on test data

final_pred = votingreg.fit(x_train, y_train).predict(X_test)
test_pred_df = pd.DataFrame(final_pred , columns=['SalePrice'])

test_id_df = pd.DataFrame(test_ID, columns=['Id'])



submission = pd.concat([test_id_df, np.expm1(test_pred_df)], axis=1)

submission.head()

submission.to_csv('submission.csv', index=False)