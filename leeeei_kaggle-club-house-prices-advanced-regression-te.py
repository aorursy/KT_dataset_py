# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# dataframe
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

# stats
from scipy import stats
from scipy.stats import norm, skew

# plotting
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns # Why sns not sbn? https://www.reddit.com/r/learnpython/comments/5oscmr/why_is_seaborn_commonly_imported_as_sns/
color = sns.color_palette()
sns.set_style('darkgrid')

# modeling
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LassoCV
from sklearn import metrics 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV, cross_val_score

# others
import warnings
warnings.filterwarnings("ignore") #ignore annoying warning (from sklearn and seaborn)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# Number of rows and columns of the dataframe
train.shape
# First and intuitive impression of what data in each column is like
train.head()
# Nullness and types of each column
train.info()
# Statistic summary of each column
# It could be more useful to inspect each column's statistic summary if necessary than the whole dataframe
train.describe()
numerical_features = train.select_dtypes(exclude='object').drop(['Id', 'SalePrice'], axis=1)
# Statistic summary of target variable
train.SalePrice.describe()
print("Skewness: %f" % train.SalePrice.skew())
print("Kurtosis: %f" % train.SalePrice.kurt())

# Plot the distribution against the normal distribution
sns.distplot(train.SalePrice , fit=norm);

# Get the fitted parameters used by the function
# mu and sigma (aka location and scale) are MLEs(Maximum Likelihood Estimate)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
(mu, sigma) = norm.fit(train.SalePrice)

# Set some parameters and show the distribution plot
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Distribution')

# Show the probability plot
# More about Q-Q plot and P-P plot: https://www.theanalysisfactor.com/anatomy-of-a-normal-probability-plot/
# The closer to normal distribution, the more aligned with y=x line
fig = plt.figure()
res = stats.probplot(train.SalePrice, plot=plt)
plt.show()
# We use the numpy fuction log1p which applies log(1+x) to all elements of the column
train.SalePrice = np.log1p(train.SalePrice)
# Check the distribution again after transformation
print("Skewness: %f" % train.SalePrice.skew())
print("Kurtosis: %f" % train.SalePrice.kurt())

# Plot the distribution against the normal distribution
sns.distplot(train.SalePrice , fit=norm);

# Get the fitted parameters used by the function
# mu and sigma (aka location and scale) are MLEs(Maximum Likelihood Estimate)
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit
(mu, sigma) = norm.fit(train.SalePrice)

# Set some parameters and show the distribution plot
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
plt.title('Distribution')

# Show the probability plot
# More about Q-Q plot and P-P plot: https://www.theanalysisfactor.com/anatomy-of-a-normal-probability-plot/
# The closer to normal distribution, the more aligned with y=x line
fig = plt.figure()
res = stats.probplot(train.SalePrice, plot=plt)
plt.show()
fig = plt.figure(figsize=(16,20))

for i in range(len(numerical_features.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.boxplot(y=numerical_features.iloc[:,i])

plt.tight_layout()
plt.show()
fig = plt.figure(figsize=(12,18))
for i in range(len(numerical_features.columns)):
    fig.add_subplot(9, 4, i+1)
    sns.scatterplot(numerical_features.iloc[:, i], train.SalePrice)
plt.tight_layout()
plt.show()
train = train.drop(train[train.GrLivArea > 4000].index)
train.shape
ntrain = train.shape[0]
ntest = test.shape[0]
test_id = test['Id']
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Id', 'SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
all_data.head()
missing_total = all_data.isnull().sum()
missing_total.drop(missing_total[missing_total == 0].index, inplace = True)
missing_total.sort_values(ascending=False, inplace = True)
missing_percent = (missing_total / len(all_data) * 100)
missing_data = pd.concat([missing_total, missing_percent, all_data[missing_total.index].dtypes], axis=1, keys=['Total', 'Percent', 'Type'])
print(missing_data)
#for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'):
#   all_data[col] = all_data[col].fillna('None')
for col in missing_data[(missing_data.Total >= 24) & (missing_data.Type == 'object')].index:
    all_data[col] = all_data[col].fillna('None')
# LotFrontage is correlated to the 'Neighborhood' feature because the LotFrontage for nearby houses will be really similar, so we fill in missing values by the median based off of Neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Why `GarageFinish` has 159 missings while `GarageType` has 157 
all_data[(all_data.GarageFinish == 'None') & (all_data.GarageType != 'None')].filter(like="Garage", axis=1)
# Why `GarageFinish` has 159 missings while `GarageArea` and `GarageCars` have only one
all_data[all_data.GarageFinish == 'None'].filter(like="Garage", axis=1)
# Why `GarageFinish` has 159 missings while `GarageArea` and `GarageCars` have only one
all_data[(all_data.GarageFinish == 'None') & (all_data.GarageArea != 0)].filter(like="Garage", axis=1)
# Imputing numeric features with 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
# Correct the `GarageType` to be consistent with other features
all_data.loc[2572, 'GarageType'] = 'None'
# Why `BsmtExposure` has 82 missings while `BsmtFinType1` has 79 
all_data[(all_data.BsmtExposure == 'None') & (all_data.BsmtFinType1 != 'None')].filter(like="Bsmt", axis=1)
# Why `BsmtExposure` has 82 missings while `TotalBsmtSF` has 1
all_data[all_data.BsmtExposure == 'None'].filter(like="Bsmt", axis=1)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
all_data[(all_data.MasVnrType == 'None') & (all_data.MasVnrArea != 0)].filter(like="MasVnr", axis=1)
all_data[all_data['MasVnrArea'] == 1] 
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data[all_data['MasVnrArea'] == 1] = 0
# Imputing with the most common seen value for these features
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

# Functional : data description says NA means typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data.groupby("Utilities").count()

# Utilities: Records are "AllPub" except for one "NoSeWa" and 2 NA.
# This feature won't help in predictive modelling. We can then safely remove it.
all_data = all_data.drop(['Utilities'], axis=1)
# Check missing values again
missing_total = all_data.isnull().sum()
missing_total.drop(missing_total[missing_total == 0].index, inplace = True)
missing_total.sort_values(ascending=False, inplace = True)
missing_percent = (missing_total / len(all_data) * 100)
missing_data = pd.concat([missing_total, missing_percent, all_data[missing_total.index].dtypes], axis=1, keys=['Total', 'Percent', 'Type'])
print(missing_data)
figure, (ax1, ax2, ax3,ax4) = plt.subplots(nrows=1, ncols=4)
figure.set_size_inches(20,10)
_ = sns.regplot(train['TotalBsmtSF'], train['SalePrice'], ax=ax1)
_ = sns.regplot(train['1stFlrSF'], train['SalePrice'], ax=ax2)
_ = sns.regplot(train['2ndFlrSF'], train['SalePrice'], ax=ax3)
_ = sns.regplot(train['TotalBsmtSF'] + train['2ndFlrSF']+train['1stFlrSF'], train['SalePrice'], ax=ax4)
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# shape        
print('Shape all_data: {}'.format(all_data.shape))
# Check the skewness of all numerical features
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for i in skewed_features:
    all_data[i] = boxcox1p(all_data[i], lam)
    
#all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data = pd.get_dummies(all_data)
print(all_data.shape)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
import xgboost as xgb
import lightgbm as lgb
# Splitting dataset back into train and test data sets
train = all_data[:ntrain]
test = all_data[ntrain:]
# Define RMSE function
def rmse_cv(model):
    # Indicate number of folds for cross validation
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
    return np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf)).mean()
# LASSO Regression Model
# This model may be very sensitive to outliers. 
# So we need to made it more robust on them. 
# For that we use the sklearn's Robustscaler() method on pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))

# Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

# Gradient Boosting Regression
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

# LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
# Model Performance
results = pd.DataFrame({
    'Model':['Lasso',
             'ElasticNet',
             'Ridge',
             'GBoost',
             'LightGBM',
             'XGBOOST'],
    'Score':[rmse_cv(lasso),
             rmse_cv(ENet),
             rmse_cv(KRR),
             rmse_cv(GBoost),
             rmse_cv(model_xgb),
             rmse_cv(model_lgb)
            ]})

sorted_result = results.sort_values(by='Score', ascending=True).reset_index(drop=True)
sorted_result
model_stacked = StackingCVRegressor(regressors=(ENet, lasso, model_lgb, GBoost), 
                               meta_regressor=model_xgb,
                               use_features_in_secondary=True)

print(rmse_cv(model_stacked))
def rmse(model):
    return np.sqrt(mean_squared_error(model.predict(train.values), y_train))
# Fitting models on the whole train set and check rmse
lasso.fit(train.values, y_train)
ENet.fit(train.values, y_train)
KRR.fit(train.values, y_train)
GBoost.fit(train.values, y_train)
model_xgb.fit(train.values, y_train)
model_lgb.fit(train.values, y_train)
model_stacked.fit(train.values, y_train)

results_train = pd.DataFrame({
    'Model':['Lasso',
             'ElasticNet',
             'Ridge',
             'GBoost',
             'LightGBM',
             'XGBOOST',
             'Stacked'],
    'Score':[rmse(lasso),
             rmse(ENet),
             rmse(KRR),
             rmse(GBoost),
             rmse(model_xgb),
             rmse(model_lgb),
             rmse(model_stacked)
            ]})

sorted_result_train = results_train.sort_values(by='Score', ascending=True).reset_index(drop=True)
sorted_result_train
# Ensemble predictions
final_prediction = np.expm1(
    0.4 * GBoost.predict(test.values) + 
    0.4 * model_stacked.predict(test.values) + 
    0.2 * model_xgb.predict(test.values) 
)
submission = pd.DataFrame()
submission['Id'] = test_id
submission['SalePrice'] = final_prediction
submission.to_csv('submission.csv',index=False)
