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
import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LassoCV
from sklearn import metrics 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

import numpy as np # linear algebra
import pandas as pd

min_val_corr = 0.4 
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test_Id = test_df['Id']

train_df.drop('Id', axis=1, inplace=True)
test_df.drop('Id', axis=1, inplace=True)

train_df.shape
test_df.shape
#train_category = train_df.dtypes[train_df.dtypes == "object"].index
#test_numeric = test_df.dtypes[test_df.dtypes != "object"].index
#test_category = test_df.dtypes[test_df.dtypes == "object"].index

#train_df[train_numeric].columns
#test_df[test_numeric].columns
#train_df[train_category].columns
#test_df[train_category].columns
print("Skewness: %f" % train_df['SalePrice'].skew())
print("Kurtosis: %f" % train_df['SalePrice'].kurt())

figure = plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
sns.distplot(train_df['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train_df['SalePrice'])
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')

plt.subplot(1,2,2)
stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
# Print out Skewness and Kurtosis
print("Skewness: %f" % train_df['SalePrice'].skew())
print("Kurtosis: %f" % train_df['SalePrice'].kurt())

figure = plt.figure(figsize=(18,10))
plt.subplot(1,2,1)
sns.distplot(train_df['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train_df['SalePrice'])
plt.ylabel('Frequency')
plt.title('SalePrice Distribution')

plt.subplot(1,2,2)
stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()
num_features = train_df.select_dtypes(exclude='object').drop(['SalePrice'], axis=1)
fig = plt.figure(figsize=(16,20))

for i in range(len(num_features.columns)):
    fig.add_subplot(9, 5, i+1)
    sns.boxplot(y=num_features.iloc[:,i])

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(12,18))
for i in range(len(num_features.columns)):
    fig.add_subplot(9, 5, i+1)
    sns.scatterplot(num_features.iloc[:, i],train_df['SalePrice'])
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3,2),(0,0))
plt.scatter(x=train_df['GrLivArea'], y=train_df['SalePrice'], color=('yellowgreen'), alpha=0.5)
plt.axvline(x=4600, color='r', linestyle='-')
plt.title('Ground living Area- Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(0,1))
plt.scatter(x=train_df['TotalBsmtSF'], y=train_df['SalePrice'], color=('red'),alpha=0.5)
plt.axvline(x=5900, color='r', linestyle='-')
plt.title('Basement Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,0))
plt.scatter(x=train_df['1stFlrSF'], y=train_df['SalePrice'], color=('deepskyblue'),alpha=0.5)
plt.axvline(x=4000, color='r', linestyle='-')
plt.title('First floor Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(1,1))
plt.scatter(x=train_df['MasVnrArea'], y=train_df['SalePrice'], color=('gold'),alpha=0.9)
plt.axvline(x=1500, color='r', linestyle='-')
plt.title('Masonry veneer Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,0))
plt.scatter(x=train_df['GarageArea'], y=train_df['SalePrice'], color=('orchid'),alpha=0.5)
plt.axvline(x=1230, color='r', linestyle='-')
plt.title('Garage Area - Price scatter plot', fontsize=15, weight='bold' )

ax1 = plt.subplot2grid((3,2),(2,1))
plt.scatter(x=train_df['TotRmsAbvGrd'], y=train_df['SalePrice'], color=('tan'),alpha=0.9)
plt.axvline(x=13, color='r', linestyle='-')
plt.title('TotRmsAbvGrd - Price scatter plot', fontsize=15, weight='bold' )
plt.show()


pos = [1298,523,297]
train_df.drop(train_df.index[pos], inplace=True)

train_df.head()
train_df.shape
# Save target value for later
y = train_df.SalePrice.values

# In order to make imputing easier, we combine train and test data
train_df.drop(['SalePrice'], axis=1, inplace=True)
dataset = pd.concat((train_df, test_df)).reset_index(drop=True)
dataset.head()
dataset.shape
missing = dataset.isnull().sum().sort_values(ascending=False)
total = dataset.isnull().count().sort_values(ascending=False)
percent = (dataset.isnull().sum() / dataset.isnull().count()).sort_values(ascending=False)
Missing_values = pd.concat([missing,percent], axis=1, keys=['Missing','Percent'])
Missing_values.head(20)
sns.set()
missing.plot(kind='bar',figsize=(20,8))

dataset[missing.index].dtypes
for col in ('FireplaceQu', 'Fence', 'Alley', 'MiscFeature', 'PoolQC', 'MSSubClass'):
    dataset[col] = dataset[col].fillna('None')

for col in ('Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'Functional', 'MSZoning', 'SaleType', 'Utilities'):
    dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    dataset[col] = dataset[col].fillna(0)
    
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    dataset[col] = dataset[col].fillna('None')
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    dataset[col] = dataset[col].fillna(0)

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    dataset[col] = dataset[col].fillna('None')
dataset.isnull().sum()[dataset.isnull().sum() > 0].sort_values(ascending=False)
dataset['MasVnrType'] = dataset['MasVnrType'].fillna('None')
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(0)
# LotFrontage is correlated to the 'Neighborhood' feature because the LotFrontage for nearby houses will be really similar, so we fill in missing values by the median based off of Neighborhood
dataset["LotFrontage"] = dataset.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
list(dataset.select_dtypes(exclude='object').columns)
# Features 'MSSubClass', 'YrSold', 'OverallCond' and 'MoSold' are supposed to be categorical features so change their type
dataset['MSSubClass'] = dataset['MSSubClass'].apply(str)
dataset['YrSold'] = dataset['YrSold'].apply(str)
dataset['MoSold'] = dataset['MoSold'].apply(str)
dataset['OverallCond'] = dataset['OverallCond'].astype(str)

# Features 'LotArea' and 'MasVnrArea' are supposed to be numerical features so change their type
dataset['LotArea'] = dataset['LotArea'].astype(np.int64)
dataset['MasVnrArea'] = dataset['MasVnrArea'].astype(np.int64)

figure, (ax1, ax2, ax3,ax4) = plt.subplots(nrows=1, ncols=4)
figure.set_size_inches(20,10)
_ = sns.regplot(train_df['TotalBsmtSF'], y, ax=ax1)
_ = sns.regplot(train_df['1stFlrSF'], y, ax=ax2)
_ = sns.regplot(train_df['2ndFlrSF'], y, ax=ax3)
_ = sns.regplot(train_df['TotalBsmtSF'] + train_df['2ndFlrSF']+train_df['1stFlrSF'], y, ax=ax4)
dataset['TotalSF']=dataset['TotalBsmtSF']  + dataset['1stFlrSF'] + dataset['2ndFlrSF']
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
figure.set_size_inches(14,10)
_ = sns.barplot(train_df['BsmtFullBath'], y, ax=ax1)
_ = sns.barplot(train_df['FullBath'], y, ax=ax2)
_ = sns.barplot(train_df['BsmtHalfBath'], y, ax=ax3)
_ = sns.barplot(train_df['BsmtFullBath'] + train_df['FullBath'] + train_df['BsmtHalfBath'] + train_df['HalfBath'], y, ax=ax4)
dataset['TotalBath']=dataset['BsmtFullBath'] + dataset['FullBath'] + (0.5*dataset['BsmtHalfBath']) + (0.5*dataset['HalfBath'])
figure, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
figure.set_size_inches(18,8)
_ = sns.regplot(train_df['YearBuilt'], y, ax=ax1)
_ = sns.regplot(train_df['YearRemodAdd'],y, ax=ax2)
_ = sns.regplot((train_df['YearBuilt']+train_df['YearRemodAdd'])/2, y, ax=ax3)
dataset['YrBltAndRemod']=dataset['YearBuilt']+dataset['YearRemodAdd']
figure, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3)
figure.set_size_inches(20,10)
_ = sns.regplot(train_df['OpenPorchSF'], y, ax=ax1)
_ = sns.regplot(train_df['3SsnPorch'], y, ax=ax2)
_ = sns.regplot(train_df['EnclosedPorch'], y, ax=ax3)
_ = sns.regplot(train_df['ScreenPorch'], y, ax=ax4)
_ = sns.regplot(train_df['WoodDeckSF'], y, ax=ax5)
_ = sns.regplot((train_df['OpenPorchSF']+train_df['3SsnPorch']+train_df['EnclosedPorch']+train_df['ScreenPorch']+train_df['WoodDeckSF']), y, ax=ax6)
dataset['Porch_SF'] = (dataset['OpenPorchSF'] + dataset['3SsnPorch'] + dataset['EnclosedPorch'] + dataset['ScreenPorch'] + dataset['WoodDeckSF'])
dataset['Has2ndfloor'] = dataset['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
dataset['HasBsmt'] = dataset['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
dataset['HasFirePlace'] =dataset['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
dataset['Has2ndFlr']=dataset['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
dataset['HasPool']=dataset['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
from sklearn.preprocessing import LabelEncoder
categorical_col = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for col in categorical_col:
    label = LabelEncoder() 
    label.fit(list(dataset[col].values)) 
    dataset[col] = label.transform(list(dataset[col].values))

print('Shape all_data: {}'.format(dataset.shape))
num_features = dataset.dtypes[dataset.dtypes != 'object'].index
skewed_features = dataset[num_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_features})
skewness.head(15)

numerical_dataset = dataset.select_dtypes(exclude='object')

for i in range(len(numerical_dataset.columns)):
    f, ax = plt.subplots(figsize=(7, 4))
    fig = sns.distplot(numerical_dataset.iloc[:,i].dropna(), rug=True, hist=False, label='UW', kde_kws={'bw':0.1})
    plt.xlabel(numerical_dataset.columns[i])
skewness = skewed_features[abs(skewed_features) > 0.75]
skewed_features = skewness.index

lam = 0.15
for i in skewed_features:
    dataset[i] = boxcox1p(dataset[i], lam)

from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error , make_scorer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# Hot-Encode Categorical features
dataset = pd.get_dummies(dataset) 

# Splitting dataset back into X and test data
X = dataset[:len(train_df)]
test = dataset[len(train_df):]

X.shape
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=0)
# Indicate number of folds for cross validation
kfolds = KFold(n_splits=5, shuffle=True, random_state=42)

# Parameters for models
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
# Lasso Model
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas = alphas2, random_state = 42, cv=kfolds))

# Printing Lasso Score with Cross-Validation
lasso_score = cross_val_score(lasso, X, y, cv=kfolds, scoring='neg_mean_squared_error')
lasso_rmse = np.sqrt(-lasso_score.mean())
print("LASSO RMSE: ", lasso_rmse)
print("LASSO STD: ", lasso_score.std())

# Training Model for later
lasso.fit(X_train, y_train)
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = alphas_alt, cv=kfolds))
ridge_score = cross_val_score(ridge, X, y, cv=kfolds, scoring='neg_mean_squared_error')
ridge_rmse =  np.sqrt(-ridge_score.mean())
# Printing out Ridge Score and STD
print("RIDGE RMSE: ", ridge_rmse)
print("RIDGE STD: ", ridge_score.std())
# Training Model for later
ridge.fit(X_train, y_train)
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
elastic_score = cross_val_score(elasticnet, X, y, cv=kfolds, scoring='neg_mean_squared_error')
elastic_rmse =  np.sqrt(-elastic_score.mean())

# Printing out ElasticNet Score and STD
print("ELASTICNET RMSE: ", elastic_rmse)
print("ELASTICNET STD: ", elastic_score.std())

# Training Model for later
elasticnet.fit(X_train, y_train)
lightgbm = make_pipeline(RobustScaler(),
                        LGBMRegressor(objective='regression',num_leaves=5,
                                      learning_rate=0.05, n_estimators=720,
                                      max_bin = 55, bagging_fraction = 0.8,
                                      bagging_freq = 5, feature_fraction = 0.2319,
                                      feature_fraction_seed=9, bagging_seed=9,
                                      min_data_in_leaf =6, 
                                      min_sum_hessian_in_leaf = 11))

# Printing out LightGBM Score and STD
lightgbm_score = cross_val_score(lightgbm, X, y, cv=kfolds, scoring='neg_mean_squared_error')
lightgbm_rmse = np.sqrt(-lightgbm_score.mean())
print("LIGHTGBM RMSE: ", lightgbm_rmse)
print("LIGHTGBM STD: ", lightgbm_score.std())
# Training Model for later
lightgbm.fit(X_train, y_train)
xgboost = make_pipeline(RobustScaler(),
                        XGBRegressor(learning_rate =0.01, n_estimators=3460, 
                                     max_depth=3,min_child_weight=0 ,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,nthread=4,
                                     scale_pos_weight=1,seed=27, 
                                     reg_alpha=0.00006))

# Printing out XGBOOST Score and STD
xgboost_score = cross_val_score(xgboost, X, y, cv=kfolds, scoring='neg_mean_squared_error')
xgboost_rmse = np.sqrt(-xgboost_score.mean())
print("XGBOOST RMSE: ", xgboost_rmse)
print("XGBOOST STD: ", xgboost_score.std())
# Training Model for later
xgboost.fit(X_train, y_train)
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, 
                                            xgboost, lightgbm), 
                               meta_regressor=xgboost,
                               use_features_in_secondary=True)

stack_score = cross_val_score(stack_gen, X, y, cv=kfolds, scoring='neg_mean_squared_error')
stack_rmse = np.sqrt(-stack_score.mean())
print("STACK RMSE: ", stack_rmse)
print("STACK STD: ", stack_score.std())

#Training Model for later
stack_gen.fit(X_train, y_train)

results = pd.DataFrame({
    'Model':['Lasso',
            'Ridge',
            'ElasticNet',
            'LightGBM',
            'XGBOOST',
            'STACK'],
    'Score':[lasso_rmse,
             ridge_rmse,
             elastic_rmse,
             lightgbm_rmse,
             xgboost_rmse,
             stack_rmse
            ]})

sorted_result = results.sort_values(by='Score', ascending=True).reset_index(drop=True)
sorted_result

f, ax = plt.subplots(figsize=(14,8))
plt.xticks(rotation='90')
sns.barplot(x=sorted_result['Model'], y=sorted_result['Score'])
plt.xlabel('Model', fontsize=15)
plt.ylabel('Performance', fontsize=15)
plt.ylim(0.10, 0.12)
plt.title('RMSE', fontsize=15)
# Predict every model
lasso_pred = lasso.predict(test)
ridge_pred = ridge.predict(test)
elasticnet_pred = elasticnet.predict(test)
lightgbm_pred = lightgbm.predict(test)
xgboost_pred = xgboost.predict(test)
stacked_pred = stack_gen.predict(test)
# Combine predictions into final predictions
final_predictions = np.expm1((0.2*elasticnet_pred) + (0.2*lasso_pred) + (0.2*ridge_pred) + 
               (0.1*xgboost_pred) + (0.1*lightgbm_pred) + (0.2*stacked_pred))
submission = pd.DataFrame()
submission['Id'] = test_Id
submission['SalePrice'] = final_predictions
submission.to_csv('submission.csv',index=False)