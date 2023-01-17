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
#Esentials
import pandas as pd
import numpy as np
import math
import os

# Statistics
from scipy import stats
from scipy.stats import norm, skew

#Visualization
#import dabl
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
%matplotlib inline

# Ignore useless warnings
import warnings
warnings.filterwarnings("ignore")

#Limiting floats output to 2 decimal points
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) 

# the below code will show all the columns of the datasets
pd.set_option('max_columns', 82)

# Scikit Learn Librraies for Model building
import sklearn.metrics as metrics
from sklearn.preprocessing import power_transform
from sklearn.metrics import mean_squared_error, r2_score

#!pip install mlxtend
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

#!pip install lightgbm
from lightgbm import LGBMRegressor

#pip install xgboost
from xgboost import XGBRegressor


#print(os.getcwd())
X = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv").set_index("Id")
Y_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv").set_index("Id")
Y_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv").set_index("Id")
print("The dimension of X(Train) Dataset is:", X.shape)
print(X.duplicated().sum(),'Duplicate rows.')
X.head()
print("The dimension of Y(Train) Dataset is:", Y_train.shape)
print(Y_train.duplicated().sum(),'Duplicate rows.')
Y_train.head()
print("The dimension of Y(Test) Dataset is:", Y_test.shape)
print(Y_test.duplicated().sum(),'Duplicate rows.')
Y_test.head()
# the below code is not working in kaggle notebook
#dabl.plot(X, target_col='SalePrice') 
sns.set_color_codes(palette='dark')
sns.distplot(X['SalePrice'] , fit=norm);

#Now plot the distribution
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(X['SalePrice'], plot=plt)
plt.show()
# Visualizing the skewness of the dataset

f4, axes4 = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
f4.subplots_adjust(hspace= .5)

X.skew().plot(ax=axes4[0]).set_title('Skewness in X dataset')
Y_train.skew().plot(ax=axes4[1]).set_title('Skewness in Y dataset')
X.info()
Y_train.info()
X.describe(percentiles= [.25, .5, .75, .99] )
X.describe(exclude = [np.number])
Y_train.describe(percentiles= [.25, .5, .75, .99])
Y_train.describe(exclude = [np.number])
print("A sample of 50 data points are taken and observed if missing data is occuring.")
msno.matrix(X.sample(50))
print("Correlation of missing values in X dataset")

msno.heatmap(X)
print("Correlation of missing values in Y dataset")

msno.heatmap(Y_train)
# Return list of columns that contain missing values in X Dataset:

def get_missing_columns(X):
    return X.columns[X.isna().any()].tolist()
print(len(get_missing_columns(X)), "columns have missing data!!")
print("\nColumns names in a list:", get_missing_columns(X) )
# Return list of columns that contain missing values in Y Dataset:

def get_missing_columns(Y_train):
    return Y_train.columns[Y_train.isna().any()].tolist()
print(len(get_missing_columns(Y_train)), "columns have missing data!!")
print("\nColumns names in a list:", get_missing_columns(Y_train))
# Graph showing Missing Values in X & Y dataset (in numbers)

X_null = [feature for feature in X.columns if (X[feature].isnull().sum() > 0)]
Y_null = [feature for feature in Y_train.columns if (Y_train[feature].isnull().sum() > 0)]

f, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
f.subplots_adjust(hspace= .5)

X[X_null].isnull().sum().plot(kind='bar', ax=axes[0]).set_title('Features with missing data in X Dataset')
Y_train[Y_null].isnull().sum().plot(kind='bar', ax=axes[1]).set_title('Features with missing data in Y Dataset')
# Graph showing Missing Values in X & Y dataset (in Percentage)

f1, axes1 = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
f1.subplots_adjust(hspace= .5)

X[X_null].isnull().mean().plot(kind='bar', ax=axes1[0]).set_title('PERCENTAGE of missing Features in X Dataset')
Y_train[Y_null].isnull().mean().plot(kind='bar', ax=axes1[1]).set_title('PERCENTAGE of missing Features in Y Dataset')
# table showing percentages of the missing values
print("The attributes in X dataset which has more than 40 % of the missing values")
X_NA = [(c, X[c].isna().mean()*100) for c in X]
X_NA = pd.DataFrame(X_NA, columns=["column_name", "percentage"])
X_NA = X_NA[X_NA.percentage > 40]
X_NA.sort_values("percentage", ascending=False)
# table showing percentages of the missing values
print("The attributes in Y dataset which has more than 40 % of the missing values")
Y_NA = [(c, Y_train[c].isna().mean()*100) for c in Y_train]
Y_NA = pd.DataFrame(Y_NA, columns=["column_name", "percentage"])
Y_NA = Y_NA[Y_NA.percentage > 40]
Y_NA.sort_values("percentage", ascending=False)
# dropping these variables 
drop_columns = ['PoolQC', 'MiscFeature', 'Alley', 'Fence','FireplaceQu']
X.drop(X[drop_columns], inplace = True, axis=1)
Y_train.drop(Y_train[drop_columns], inplace = True, axis=1)
X.shape, Y_train.shape
X_cat = X.select_dtypes(include=['object'])
X_NULL = pd.DataFrame(X_cat.isna().sum(), columns=["Missing"])
X_NULL.sort_values("Missing", ascending=False).head(11)
Y_cat = Y_train.select_dtypes(include=['object'])
Y_NULL = pd.DataFrame(Y_cat.isna().sum(), columns=["Missing"])
Y_NULL.sort_values("Missing", ascending=False).head(17)
X_col = ['GarageCond','GarageQual','GarageFinish','GarageType','BsmtExposure','BsmtFinType2','BsmtCond','BsmtFinType1','BsmtQual','MasVnrType','Electrical']
X[X_col] = X[X_col].fillna(X.mode().iloc[0])

Y_col = ['GarageCond','GarageQual','GarageFinish','GarageType', 'BsmtCond','BsmtExposure', 'BsmtQual', 'BsmtFinType1','BsmtFinType2','MasVnrType','MSZoning','Functional','Utilities','KitchenQual','Exterior1st','Exterior2nd','SaleType']
Y_train[Y_col] = Y_train[Y_col].fillna(Y_train.mode().iloc[0])
X_num = X.select_dtypes(exclude=['object'])
X_NULL_num = pd.DataFrame(X_num.isna().sum(), columns=["Missing"])
X_NULL_num.sort_values("Missing", ascending=False).head(3)
Y_num = Y_train.select_dtypes(exclude=['object'])
Y_NULL_num = pd.DataFrame(Y_num.isna().sum(), columns=["Missing"])
Y_NULL_num.sort_values("Missing", ascending=False).head(11)
X_col1 = ['LotFrontage','GarageYrBlt','MasVnrArea']
X[X_col1] = X[X_col1].fillna(X.mean().iloc[0])

Y_col1 = ['LotFrontage','GarageYrBlt','MasVnrArea','BsmtHalfBath','BsmtFullBath','TotalBsmtSF','GarageCars','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','GarageArea']
Y_train[Y_col1] = Y_train[Y_col1].fillna(Y_train.mean().iloc[0])
#X Dataset
outliers_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']

for feature in outliers_features:
    IQR = X[feature].quantile(0.75) - X[feature].quantile(0.25)
    lower_boundary = X[feature].quantile(0.25) - (IQR*3)
    upper_boundary = X[feature].quantile(0.75) + (IQR*3)
    print(feature, lower_boundary, upper_boundary)
    
for feature in outliers_features:
    IQR = X[feature].quantile(0.75) - X[feature].quantile(0.25)
    lower_boundary = X[feature].quantile(0.25) - (IQR*3)
    upper_boundary = X[feature].quantile(0.75) + (IQR*3)
    X.loc[X[feature]<=lower_boundary, feature] = lower_boundary
    X.loc[X[feature]>=upper_boundary, feature] = upper_boundary
    
X[outliers_features].describe()    
# Y Dataset
outliers_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']

for feature in outliers_features:
    IQR = Y_train[feature].quantile(0.75) - Y_train[feature].quantile(0.25)
    lower_boundary = Y_train[feature].quantile(0.25) - (IQR*3)
    upper_boundary = Y_train[feature].quantile(0.75) + (IQR*3)
    print(feature, lower_boundary, upper_boundary)
    
for feature in outliers_features:
    IQR = Y_train[feature].quantile(0.75) - Y_train[feature].quantile(0.25)
    lower_boundary = Y_train[feature].quantile(0.25) - (IQR*3)
    upper_boundary = Y_train[feature].quantile(0.75) + (IQR*3)
    Y_train.loc[Y_train[feature]<=lower_boundary, feature] = lower_boundary
    Y_train.loc[Y_train[feature]>=upper_boundary, feature] = upper_boundary
    
Y_train[outliers_features].describe()    
corr_plot = X.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr_plot, vmax= 0.5, center=0 ,cmap= 'viridis' ,linewidths= .9 ,linecolor='white')
corr_features = corr_plot.index[abs(corr_plot['SalePrice']) < 0.3]
corr_features
X.drop(['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'KitchenAbvGr', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'EnclosedPorch' ,'MiscVal', 'MoSold', 'YrSold'], axis=1, inplace=True)
Y_train.drop(['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'HalfBath', 'KitchenAbvGr', 'KitchenAbvGr', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'EnclosedPorch' ,'MiscVal', 'MoSold', 'YrSold'], axis=1, inplace=True)
X.shape, Y_train.shape
Y = pd.concat([Y_train, Y_test], axis=1)
HP = pd.concat([X,Y], axis=0)
print('Dimension of HP dataset is',HP.shape)
HP.head()
HP = pd.get_dummies(HP)
print('Dimension of HP dataset is',HP.shape)
HP.head()
# Remove any duplicated column names
HP = HP.loc[:,~HP.columns.duplicated()]
#Log Transforming the dataset

cols_skew = [col for col in HP if '_2num' in col or '_' not in col]
HP[cols_skew].skew().sort_values()
cols_unskew = HP[cols_skew].columns[abs(HP[cols_skew].skew()) > 1]
for col in cols_unskew:
    HP[col] = np.log1p(HP[col])
HP.head()    
X = HP.iloc[:1460,:]
Y = HP.iloc[:1459,:]

X_train = X.drop('SalePrice', axis=1) #fit
X_test = X['SalePrice']               #fit         
y_train = Y.drop('SalePrice', axis=1) #predict
y_test = Y['SalePrice']               #compare


print("The dimension of X_train is:", X_train.shape)
print("The dimension of X_test is:", X_test.shape)
print("The dimension of Y_train is:", y_train.shape)
print("The dimension of Y_test is:", y_test.shape)
reg = RandomForestRegressor(n_estimators= 700,
                          max_depth= 20,
                          min_samples_split= 15,
                          min_samples_leaf= 1,
                          max_features= 'auto',
                          oob_score = True,
                          random_state = 200)

reg.fit(X_train, X_test)
RFR_y_pred = reg.predict(y_train)

print('The Out-of-bag score (oob) for Random Forest Regressor Model is',round((reg.oob_score_),2))
print('Root Mean Squared error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, RFR_y_pred))))
print('R2 score: ', round(r2_score(y_test, RFR_y_pred),2))
GBoost = GradientBoostingRegressor(n_estimators= 500,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=200)

GBoost.fit(X_train, X_test)
GBoost_y_pred = GBoost.predict(y_train)

print('Root Mean Squared error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, GBoost_y_pred))))
print('R2 score: ', round(r2_score(y_test, GBoost_y_pred),2))
lgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators= 100,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=200)

lgbm.fit(X_train, X_test)
lgbm_y_pred = lgbm.predict(y_train)

print('Root Mean Squared error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, lgbm_y_pred))))
print('R2 score: ', round(r2_score(y_test, lgbm_y_pred),2))
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators= 500,
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
                       random_state=200)

xgboost.fit(X_train, X_test)
xgboost_y_pred = xgboost.predict(y_train)

print('Root Mean Squared error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, xgboost_y_pred))))
print('R2 score: ', round(r2_score(y_test, xgboost_y_pred),2))
stack_gen = StackingCVRegressor(regressors=(reg, GBoost, lgbm, xgboost),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

stack_gen.fit(X_train, X_test)
stack_gen_y_pred = xgboost.predict(y_train)

print('Root Mean Squared error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, stack_gen_y_pred))))
Predict = (RFR_y_pred*.2 + GBoost_y_pred*.2 + lgbm_y_pred*.2 + xgboost_y_pred*.2 + stack_gen_y_pred*.2)

print('Root Mean Squared error: ' + str(math.sqrt(metrics.mean_squared_error(y_test, Predict))))
print('R2 score: ', round(r2_score(y_test, Predict),2))
# Final Predicted Dataset to be uploaded:
sub = pd.DataFrame()
sub['SalePrice'] = np.expm1(Predict)

sub.to_csv('submission.csv', index=False)
fig, ax = plt.subplots()
ax.scatter(y_test, Predict)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()
plt.savefig('Actual VS Predicted.png')