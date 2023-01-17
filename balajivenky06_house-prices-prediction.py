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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.sample(6)
train.info()
plt.scatter(train.GrLivArea,train.SalePrice, c="blue", marker = "s")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()

#Per check, 2 outliers which has greater living area but sold at very less price. Which needs to be removed
train = train[train.GrLivArea < 4500]
plt.scatter(train.LotArea,train.SalePrice, c="blue", marker = "s")
plt.xlabel("LotArea")
plt.ylabel("SalePrice")
plt.show()

#Similarly LotArea>150000 consider as outlier and remove from dataset
train = train[train.LotArea < 150000]
#Analysis on Target Variable
from scipy import stats
from scipy.stats import norm

sns.distplot(train['SalePrice'], fit= norm);

(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Per check SalePrice is highly right skewed with large kurtosis


#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
#In order for better prediction, we need to fit the target variable into Normal Dist, hence we go for log transformation
train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#Now, the target variable is more or less similar to Norm Dist...
#Data Cleaning Starts here
ntrain = train.shape[0]
ntest = test.shape[0]
ytrain = train.SalePrice.values
data = pd.concat((train, test)).reset_index(drop = True)
data.drop(['SalePrice'], axis = 1, inplace = True)
data.shape
print((data.values == 'Abnorml').sum())
col_idx = pd.np.argmax(data.values == 'Abnorml', axis=1).max()
data.iloc[:, col_idx].value_counts()
print((ytrain == 'Abnorml'))
missing = []
for col in data:
    count = data[col].isnull().sum(axis = 0)
    if count:
        missing.append(col)
        print("%s : %d" %(col, count))
data_na = (data.isnull().sum() / len(data))*100
data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending = False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' : data_na})
missing_data
f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=data_na.index, y = data_na)
plt.xlabel('Features')
plt.ylabel("% of NAN Values")
corrmat = train.corr()
corrmat.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corrmat.SalePrice)
plt.subplots(figsize=(15,12))
sns.heatmap(corrmat)
#PoolQC - Indicates houses with No POOL, hence for missing values we will fill with NONE
data['PoolQC'] = data['PoolQC'].fillna('None')
#MiscFeature - Similar to above no Miscellaneous Feature available for that house, hence fill NONE
data['MiscFeature'] = data['MiscFeature'].fillna('None')
#Alley - NA means no Passage access
data['Alley'] = data['Alley'].fillna('None')
#Fence - NA indicates no fence , hence NONE
data['Fence'] = data['Fence'].fillna('None')
#FirePlaceQu - NA means no seperate FirePlace available, hence NONE
data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
#Frontage is numerical data, it has high chance depend on neighbors area, hence we go for median imputation method
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median))
#Replace NA with None since for categorical data
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    data[col] = data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    data[col] = data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    data[col] = data[col].fillna(0)
#NA most likely means no masonry veneer for these houses. We can fill 0 for the area and None for the type.
data["MasVnrType"] = data["MasVnrType"].fillna("None")
data["MasVnrArea"] = data["MasVnrArea"].fillna(0)
data['MSZoning'] = data['MSZoning'].fillna(data['MSZoning'].mode()[0])
#Utilities - This feature doesnt help in modelling, hence removed from data
data = data.drop(['Utilities'], axis = 1)
data["Functional"] = data["Functional"].fillna("Typ")
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
data['KitchenQual'] = data['KitchenQual'].fillna(data['KitchenQual'].mode()[0])
data['Exterior1st'] = data['Exterior1st'].fillna(data['Exterior1st'].mode()[0])
data['Exterior2nd'] = data['Exterior2nd'].fillna(data['Exterior2nd'].mode()[0])
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode()[0])
data['MSSubClass'] = data['MSSubClass'].fillna("None")
# Differentiate numerical features (minus the target) and categorical features
categorical_features = train.select_dtypes(include = ['object']).columns
numerical_features = train.select_dtypes(exclude = ['object']).columns
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(data[c].values))
    data[c] = lbl.transform(list(data[c].values))
data['TotalSF'] = data['TotalBsmtSF']+data['1stFlrSF']+data['2ndFlrSF']
from scipy.stats import kurtosis
from scipy.stats import skew

data_numeric = data.dtypes[data.dtypes != 'object'].index
data_skew = data[data_numeric].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)

skewness = pd.DataFrame({'Skew': data_skew})
skewness
skewness = skewness[abs(skewness)>0.75]
from scipy.special import boxcox1p
skewness_features = skewness.index
lam = 0.15
for fld in skewness_features:
  data[fld] = boxcox1p(data[fld], lam)  
train = data[:ntrain]
test = data[:ntest]
data.info()
numeric =  data.describe().columns
#We will apply K-Fold Cross Validation for each training models

from sklearn.model_selection import KFold, cross_val_score, train_test_split

n_folds = 5
def rmse_kfold(model):
    kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train[numeric].values, ytrain, scoring = "neg_mean_squared_error", cv = kf ))
    return(rmse)
from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
lm = linear_model.LinearRegression()
score = rmse_kfold(lm)
print('\n OLS Score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmse_kfold(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
enet = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005, l1_ratio=.9,random_state = 1))
score = rmse_kfold(enet)
print("\nElastic Net Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.kernel_ridge import KernelRidge
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmse_kfold(KRR)
print("\nKRR Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmse_kfold(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
import xgboost as xgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmse_kfold(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#Having issue with install xgboost package
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmse_kfold(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.metrics import mean_squared_error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
model_lgb.fit(train[numeric], ytrain)
lgb_train_pred = model_lgb.predict(train[numeric])
lgb_pred = np.expm1(model_lgb.predict(test[numeric].values))
print(rmsle(ytrain, lgb_train_pred))
print(lgb_pred)
