import pandas as pd
 


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
train.head()
#explore data
train.info()
#save id
train_ID = train['Id']
test_ID = test['Id']
#drop id
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
train.head()
#check saleprice(target value)
sns.distplot(train['SalePrice']);
train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice']);
ntrain = train.shape[0] # number of training
ntest = test.shape[0] #number of test
y_train = train.SalePrice.values #value of saleprice

#concat train/test data
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data.shape
#check null values 
all_na = (all_data.isnull().sum() / len(all_data)) * 100
all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_na})
missing_data.head(20)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_na.index, y=all_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
#according to dic, Nan value of the pool means no oool in the house 
all_data["PoolQC"].value_counts()
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["PoolQC"].value_counts()
#according to dic, Nan value of the Miscfeature means no Miscellaneous feature in the house 
all_data["MiscFeature"].value_counts()
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["MiscFeature"].value_counts()
#according to dic, Nan value of the Alley means no Alley access in the house 
all_data["Alley"].value_counts()
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Alley"].value_counts()
#according to dic, Nan value of the Fence means no Fence in the house 
all_data["Fence"].value_counts()
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["Fence"].value_counts()
#according to dic, Nan value of the FireplaceQuality means no Fireplace in the house 
all_data["FireplaceQu"].value_counts()
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["FireplaceQu"].value_counts()
#LotFrontage: Linear feet of street connected to property
#fill with medium of neighborhood's LotFrontage
all_data["LotFrontage"].describe()
all_data["LotFrontage"].fillna(all_data.groupby("Neighborhood")["LotFrontage"].transform("median"), inplace=True)

all_data[["Neighborhood","LotFrontage"]].head(30)
#GarageYrBlt: Year garage was built
all_data[['GarageYrBlt','GarageType','GarageCars','GarageArea','GarageQual','GarageCond','GarageFinish']].head() 
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna("None")
all_data['GarageType'] = all_data['GarageType'].fillna("None")
all_data['GarageQual'] = all_data['GarageQual'].fillna("None")
all_data['GarageCond'] = all_data['GarageCond'].fillna("None")
all_data['GarageFinish'] = all_data['GarageFinish'].fillna("None")
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)

all_data[['GarageYrBlt','GarageType','GarageCars','GarageArea','GarageQual','GarageCond','GarageFinish']].head() 
#check null values..
all_data[['BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond','BsmtQual','BsmtFullBath','BsmtHalfBath','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF']].head(20) 

#if one col is null, all of these columns are null

all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna("None")
all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna("None")
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna("None")
all_data['BsmtCond'] = all_data['BsmtCond'].fillna("None")
all_data['BsmtQual'] = all_data['BsmtQual'].fillna("None")
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)

 
all_data[['BsmtFinType2','BsmtExposure','BsmtFinType1','BsmtCond','BsmtQual','BsmtFullBath','BsmtHalfBath','TotalBsmtSF']].head(20) 


# 0 MasVnrArea means no Masonry veneer
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['MasVnrType'] = all_data['MasVnrType'].fillna("None")
 
all_data['MasVnrArea'].value_counts() 
all_data['MasVnrType'].value_counts() 
#only one null value of electrical so..
all_data['Electrical'].value_counts() #SBrkr is most frequent value so we fill null as SBrkr

all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr")
all_data['MSZoning'].value_counts()
#fill Nan with RL which is most frequent value
all_data['MSZoning'] = all_data['MSZoning'].fillna("RL")
all_data['MSZoning'].value_counts()
all_data['Utilities'].value_counts()
#fill Nan with Allpub which is most frequent value 
all_data['Utilities'] = all_data['Utilities'].fillna("AllPub")
all_data['Utilities'].value_counts()
all_data['Functional'].value_counts()
#fill Nan with Typ which is most frequent value
all_data['Functional'] = all_data['Functional'].fillna("Typ")
all_data['Functional'].value_counts()
all_data['SaleType'].value_counts()
all_data['SaleType'] = all_data['SaleType'].fillna("WD")
all_data['KitchenQual'].value_counts()
all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA")
all_data['Exterior1st'].value_counts()
all_data['Exterior2nd'].value_counts()
all_data['Exterior1st'] = all_data['Exterior1st'].fillna("VinylSd")
#all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])  같은 코드 
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna("VinylSd")
#check null values 
all_na = (all_data.isnull().sum() / len(all_data)) * 100
all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_na})
missing_data.head(20)
all_data.head(10)
all_data.describe() #we will deal with scale soon 
all_data["Utilities"].value_counts()
all_data.drop('Utilities', axis=1, inplace=True)
all_data.head()
corr = train.corr(method='pearson').drop(['SalePrice']).sort_values('SalePrice', ascending=False)['SalePrice']
corr 
all_data.head()
all_data.drop('PoolArea', axis=1, inplace=True)
all_data.drop('MoSold', axis=1, inplace=True)
all_data.drop('3SsnPorch', axis=1, inplace=True)
all_data.drop('BsmtFinSF2', axis=1, inplace=True)
all_data.drop('BsmtHalfBath', axis=1, inplace=True)
all_data.drop('MiscVal', axis=1, inplace=True)
all_data.drop('LowQualFinSF', axis=1, inplace=True)
all_data.drop('YrSold', axis=1, inplace=True)
all_data.drop('OverallCond', axis=1, inplace=True)
all_data.drop('MSSubClass', axis=1, inplace=True)
all_data.shape #delete 10 cols
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew 
#get the numeric values
numeric_features = all_data.dtypes[all_data.dtypes != "object"].index
numeric_features
# Check the skew of all numerical features
skewed_feats = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness
skewness = skewness[abs(skewness)>0.5]
all_data[skewness.index] = np.log1p(all_data[skewness.index])

# Check the skew of all numerical features

skewed_feats = all_data[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness

#from sklearn.preprocessing import minmax_scale

#df['col_name']= minmax_scale(df['col_name'], axis=0, copy=True)
all_data.head()
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
all_data.shape
all_data = pd.get_dummies(all_data)
print(all_data.shape)
all_data.head()
train = all_data[:ntrain]
test = all_data[ntrain:]
#all_data[numeric_features].std()
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV
#Validation function
n_folds = 10

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=3))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5) #kernel = 'rbf' , 'sigmoid' 
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
# A parameter grid for XGBoost
#params = {'min_child_weight':[i/10.0 for i in range(5,18)], 'gamma':[i/100.0 for i in range(3,6)],  
#'subsample':[i/10.0 for i in range(4,9)], 'colsample_bytree':[i/10.0 for i in range(4,8)], 'max_depth': [2,3,4]}

# Initialize XGB and GridSearch
#xgb = XGBRegressor(nthread=-1) 

#grid = GridSearchCV(xgb, params)
#grid.fit(train, y_train)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_svr = SVR(C=1, cache_size=200, coef0=0, degree=3, epsilon=0.0, gamma='auto',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

 

#params = {'gamma' :[i/100.0 for i in range(0,11)],  
#          'coef0':[0, 0.1, 0.5, 1], 'C' :[0.1, 0.2, 0.5, 1], 'epsilon':[i/10.0 for i in range(0,6)]}

params = {'coef0':[0, 0.1, 0.5, 1], 'C' :[0.1, 0.2, 0.5, 1], 'epsilon':[i/10.0 for i in range(0,6)]}


#model_svr = SVR()
#grid_search = GridSearchCV(model_svr, params, cv=10, scoring='neg_mean_squared_error')
#grid_search.fit(train, y_train)

#grid_search.best_estimator_
regr = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features=150, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=90, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
 
param_grid = [
    {'n_estimators': [3, 10, 30, 60, 90], 'max_features': [50,100,150,200,250,300]},
    {'bootstrap': [True], 'n_estimators': [3, 10, 30, 60, 90], 'max_features': [50,100,150,200,250]},
]

#forest_reg = RandomForestRegressor()
#grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
#grid_search.fit(train, y_train)

#grid_search.best_estimator_
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
score = rmsle_cv(model_svr)
print("SVR score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

score = rmsle_cv(regr)
print("SVR score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
#define a rmsle evaluation function

def rmsle(y, y_pred): 
    return np.sqrt(mean_squared_error(y, y_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
GBoost.fit(train,y_train)
GB_train_pred = GBoost.predict(train)
GB_pred = np.expm1(GBoost.predict(test.values))
print(rmsle(y_train, GB_train_pred))
ENet.fit(train,y_train)
ENet_train_pred = ENet.predict(train)
ENet_pred = np.expm1(ENet.predict(test.values))
print(rmsle(y_train, ENet_train_pred))
ensemble = xgb_pred*0.25 + lgb_pred*0.25 + GB_pred*0.5  
#sub = pd.DataFrame()
#sub['Id'] = test_ID
#sub['SalePrice'] = ensemble
#sub.to_csv('submission.csv',index=False)
#train["SaleType"].value_counts()
