# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from scipy.stats.stats import pearsonr

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.shape, test.shape
train.columns
from scipy.stats import skew, skewtest, norm
sns.distplot(train['SalePrice'], fit=norm)
# Making the Sale price Normally distributed
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
plt.scatter(train['GrLivArea'], train['SalePrice'], marker = "s")
plt.title("Sales Price Distribution with respect to Area of house")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()
train = train[train['GrLivArea'] < 4500]
# We prefered only those data which has 'Normal' Sale Condition
train = train[train['SaleCondition'] == 'Normal']
# After reomving the inwanted data i.e. outlier
plt.scatter(train['GrLivArea'], train['SalePrice'], c = "blue", marker = "s")
plt.title("Looking for outliers")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()
corrmatrix = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmatrix, vmax=.8, square=True)
k = 11
cols = corrmatrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)
train.drop("TotRmsAbvGrd", axis = 1, inplace = True)
test.drop("TotRmsAbvGrd", axis = 1, inplace = True)
train.drop("GarageYrBlt", axis = 1, inplace = True)
test.drop("GarageYrBlt", axis = 1, inplace = True)
train.drop("GarageArea", axis = 1, inplace = True)
test.drop("GarageArea", axis = 1, inplace = True)
pd.set_option('display.max_rows', 500)
for i in train.columns:
    print(train[i].isnull().value_counts())
ntrain = train.shape[0]
ntest = test.shape[0]

# Drop the 'SalePrice' from the train data.
y = train['SalePrice']

train.drop("SalePrice", axis = 1, inplace = True)
comb_data = pd.concat([train,test]).reset_index(drop=True)
comb_data.shape
print("Combined size is : {}".format(comb_data.shape))
# Columns with missing values in them out of 1460 rows
pd.set_option('display.max_columns', 500)
null_columns=train.columns[comb_data.isnull().any()]
result = comb_data[null_columns].isnull().sum().sort_values(ascending = False)
percent = (comb_data.isnull().sum()/comb_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([result, percent], axis=1, keys=['Total', 'Percent'], sort = True).sort_values(by = 'Percent', ascending=False)
missing_data.head(20)
# Looking at the above Output we can see top four row i.e ['PoolQC', 'MiscFeature', 'Alley', 'Fence'] Consist more than 80% of data empty.
# So we can Delete or drop from our data...
comb_data.drop("PoolQC", axis = 1, inplace = True)
comb_data.drop("MiscFeature", axis = 1, inplace = True)
comb_data.drop("Alley", axis = 1, inplace = True)
comb_data.drop("Fence", axis = 1, inplace = True)
# "FireplaceQu" mean no fire place is availabel.
comb_data["FireplaceQu"] = comb_data["FireplaceQu"].fillna("None")
comb_data["LotFrontage"].fillna(0, inplace=True)
comb_data['LotFrontage'].isnull().sum()#.value_counts()
train['FireplaceQu'].value_counts()
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    comb_data[col] = comb_data[col].fillna('None')
comb_data.columns
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    comb_data[col] = comb_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    comb_data[col] = comb_data[col].fillna('None')
comb_data["MasVnrType"] = comb_data["MasVnrType"].fillna("None")
comb_data["MasVnrArea"] = comb_data["MasVnrArea"].fillna(0)
comb_data['MSZoning'] = comb_data['MSZoning'].fillna(comb_data['MSZoning'].mode()[0])
comb_data = comb_data.drop(['Utilities'], axis=1)
comb_data["Functional"] = comb_data["Functional"].fillna("Typ")
comb_data['Electrical'] = comb_data['Electrical'].fillna(comb_data['Electrical'].mode()[0])
# Same as above this column also contain only one value as missing value.
comb_data['KitchenQual'] = comb_data['KitchenQual'].fillna(comb_data['KitchenQual'].mode()[0])
comb_data['Exterior1st'] = comb_data['Exterior1st'].fillna(comb_data['Exterior1st'].mode()[0])
comb_data['Exterior2nd'] = comb_data['Exterior2nd'].fillna(comb_data['Exterior2nd'].mode()[0])
comb_data['SaleType'] = comb_data['SaleType'].fillna(comb_data['SaleType'].mode()[0])
comb_data['MSSubClass'] = comb_data['MSSubClass'].fillna("None")

comb_data["GarageCars"].fillna(0, inplace=True)
comb_data_na = (comb_data.isnull().sum() / len(comb_data)) * 100
comb_data_na = comb_data_na.drop(comb_data_na[comb_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :comb_data_na})
missing_data.head()
# Lets check which are numerical and categorical columns
comb_data.info()
#MSSubClass=The building class
comb_data['MSSubClass'] = comb_data['MSSubClass'].apply(str)

#Changing OverallCond into a categorical variable
comb_data['OverallCond'] = comb_data['OverallCond'].astype(str)

#Year and month sold are transformed into categorical features.
comb_data['YrSold'] = comb_data['YrSold'].astype(str)
comb_data['MoSold'] = comb_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(comb_data[c].values)) 
    comb_data[c] = lbl.transform(list(comb_data[c].values))

# shape        
print('Shape all_data: {}'.format(comb_data.shape))
comb_data.columns
# Adding total sqfootage feature 
comb_data['TotalSF'] = comb_data['TotalBsmtSF'] + comb_data['1stFlrSF'] + comb_data['2ndFlrSF']
numeric_feats = comb_data.dtypes[comb_data.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = comb_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)
comb_data.shape
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    comb_data[feat] = boxcox1p(comb_data[feat], lam)
comb_data = pd.get_dummies(comb_data)
print(comb_data.shape)
train = comb_data[:ntrain]
test = comb_data[ntrain:]
test.head()
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
# Applying Cross Validation function
# We have use cross-validation function of Sklearn. 

n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0004, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.kernel_ridge import KernelRidge
KRR = KernelRidge(alpha=0.7, kernel='polynomial', degree=2, coef0=7.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
import xgboost as xgb

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.03, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.04, n_estimators=750,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

class  AveragingModels(BaseEstimator,  RegressorMixin,  TransformerMixin):
    def  __init__(self,  models):
                 self.models  =  models
        
                  # we define clones of the original models to fit the data in# we define clones of the  
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
#stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)

#score = rmsle_cv(stacked_averaged_models)
#print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
averaged_models.fit(train.values, y)
average_train_pred = averaged_models.predict(train.values)
stacked_pred = np.expm1(averaged_models.predict(test.values))
print(rmsle(y, average_train_pred))
model_xgb.fit(train, y)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y, xgb_train_pred))
model_lgb.fit(train, y)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y, lgb_train_pred))
print('RMSLE score on train data:')
print(rmsle(y,average_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
sub_1 = pd.DataFrame()
sub_1['Id'] = test_ID
sub_1['SalePrice'] = ensemble
sub_1.to_csv('submission.csv',index=False)
sub_1.head()
