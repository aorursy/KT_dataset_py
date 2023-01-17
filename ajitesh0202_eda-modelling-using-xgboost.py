import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm,skew
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
train.head()
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id",axis=1,inplace=True) # Dropping TrainId 
test.drop("Id",axis=1,inplace=True)  # Dropping TestId
# Detecting Outliners within the dataset
fig,ax = plt.subplots()
ax.scatter(x=train['GrLivArea'],y=train['SalePrice'])
plt.ylabel('SalePrice',fontsize=13)
plt.xlabel('GrLivArea',fontsize=13)
plt.show()
#Removing outliers in the dataset
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig ,ax= plt.subplots()
ax.scatter(train['GrLivArea'],train['SalePrice'])
plt.xlabel('GrLivArea',fontsize=12)
plt.ylabel('SalePrice',fontsize=13)
plt.show()
# Plotting Univariate distribution using Distplot

sns.distplot(train['SalePrice'],fit=norm); 

'''
Fitting a normal distribution 
mu  - mean of the normal distribution
sigma - standard deviatio of the normal distribution

'''

(mu,sigma) = norm.fit(train['SalePrice']) 

plt.legend(['Normal dist . ($\mu=$ {:2f} and $\sigma = $ {:2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()

'''
Plotting the probability plot (QQ-Plot):
The probability plot is a graphical technique for assessing whether or 
not a data set follows a given distribution such as the normal or Weibull.
'''
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()

'''

Log transformation of data : Make highly skewed dataset less skewed 
making it easy for parametric tests 

'''
train['SalePrice'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'],fit=norm)
(mu,sigma)= norm.fit(train['SalePrice'])
plt.legend(['Normal dist . ($\mu=$ {:2f} and $\sigma=$ {:2f})'.format(mu,sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'],plot=plt)
plt.show()
ntrain  = train.shape[0]
ntest   = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train,test)).reset_index(drop=True)
all_data.drop(['SalePrice'],axis=1,inplace=True)
all_data.shape
'''
Computing the Null Values
The Ratio of Missing Data
'''
all_data_na = (all_data.isnull().sum()/len(all_data))*100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ration' : all_data_na})
missing_data.head(20)
'''
Plotting the missing value by missing value ratio
'''
f,ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index,y=all_data_na)
plt.xlabel('Features',fontsize=15)
plt.ylabel('Percent of missing values',fontsize=15)
plt.title('Percent missing data by feature',fontsize=15)
'''
Correlatio Matrix gives an idea about how related values are there in the dataset
'''

corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat,vmax=0.9,square=True)
all_data['PoolQC'] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data.dtypes
'''
Label Encoding for working with Categorical values
'''
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))

print(all_data.shape)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF']+all_data['2ndFlrSF']
#Numerical Features in the dataset
numeric_feats  = all_data.dtypes[all_data.dtypes!="object"].index
'''
In statistics, 
skewness is a measure of the asymmetry of the probability distribution 
of a random variable about its mean.
'''
skewed_feats = all_data[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' : skewed_feats})
skewness.head(10)
all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test  = all_data[ntrain:]
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error#
import xgboost as xgb
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
np.where(all_data.values >= np.finfo(np.float64).max)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=12000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
#RFR
'''regr.fit(train,y_train)
rg_train_pred = regr.predict(train)
rg_pred = np.expm1(regr.predict(test))
print(rmsle(y_train, rg_train_pred))'''
#XGB
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = xgb_pred
sub.to_csv('submission.csv',index=False)
print(sub.head())


