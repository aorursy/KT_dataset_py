import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#print(os.listdir("../input"))

import numpy as np
import pandas as pd
import matplotlib

from scipy.stats import skew
from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, cv, DMatrix, train

import lightgbm as lgb

from sklearn.base import BaseEstimator, RegressorMixin

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 5))
    return(round(rmse.mean(), 5))
#load data
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
#df_train = pd.read_csv("data/train.csv")
#df_test = pd.read_csv("data/test.csv")

#drop outliers how described in [1]
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)

df_train_test = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],
                      df_test.loc[:,'MSSubClass':'SaleCondition']))

TRAIN_SIZE = df_train.shape[0]
#log transform the target because of (R)MSLE metric:
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

#MSSubClass to str
df_train_test['MSSubClass'] = df_train_test['MSSubClass'].apply(lambda x: str(x))

#log transform skewed numeric features:
numeric_feats = df_train_test.dtypes[df_train_test.dtypes != "object"].index

#find skewed features and log tranform of them to normalize
skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

df_train_test[skewed_feats] = np.log1p(df_train_test[skewed_feats])
#drop features with one large class (>98%)
df_train_test.drop(columns=['Street'], inplace=True)
df_train_test.drop(columns=['Utilities'], inplace=True)
df_train_test.drop(columns=['Condition2'], inplace=True)
df_train_test.drop(columns=['RoofMatl'], inplace=True)
df_train_test.drop(columns=['Heating'], inplace=True)

#new feature - total area of house as a sum of all floors area
df_train_test['TotalSF'] = df_train_test['TotalBsmtSF'] + df_train_test['1stFlrSF'] + df_train_test['2ndFlrSF']

#price per feet engineering:
#cost of 1 square feet of living area per house by Neighborhood groups
#Its a kind of mean encoding (or likehood encoding)
df_train['DolPerFeetLiv'] = df_train['SalePrice']/df_train['GrLivArea']
data = pd.concat([df_train['Neighborhood'], df_train['DolPerFeetLiv']], axis=1)
cost_per_district = data.groupby('Neighborhood')['DolPerFeetLiv'].mean()
df_train_test['DolPerFeetNeigLiv'] = df_train_test['Neighborhood'].apply(lambda x: cost_per_district[x])

#cost of 1 square feet of lot area per house by Neighborhood groups
df_train['DolPerFeetLot'] = df_train['SalePrice']/df_train['LotArea']
data = pd.concat([df_train['Neighborhood'], df_train['DolPerFeetLot']], axis=1)
cost_per_district = data.groupby('Neighborhood')['DolPerFeetLot'].mean()
df_train_test['DolPerFeetNeigLot'] = df_train_test['Neighborhood'].apply(lambda x: cost_per_district[x])
#dummy - encoding
df_train_test_dummies = pd.get_dummies(df_train_test)

#filling NA's with the mean of the column:
df_train_test_dummies = df_train_test_dummies.fillna(df_train_test_dummies.mean())

#train test splitting
X = df_train_test_dummies[:TRAIN_SIZE]
X_test = df_train_test_dummies[TRAIN_SIZE:]
y = df_train.SalePrice

print('shape for Lasso: ', X.shape)
#Lasso tune hyperparams
#107 vars, rsmse = 0.11004
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X, y)
lasso_rmse = rmse_cv(model_lasso)
coef = pd.Series(model_lasso.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

print(rmse_cv(model_lasso).mean())

model_lasso.fit(X, y)
y_lasso = model_lasso.predict(X_test)
y_lasso = np.expm1(y_lasso)
#feature engineering for boosting
def Qual_to_num(x):
    if x == 'Ex':
        return 4
    elif x == 'Gd':
        return 3
    elif x == 'TA':
        return 2
    else:
        return 1  

df_train_test['KitchenQualNum'] = df_train_test.KitchenQual.apply(Qual_to_num)
df_train_test['HeatingQCNum'] = df_train_test.HeatingQC.apply(Qual_to_num)

df_train_test_dummies = pd.get_dummies(df_train_test)

#filling NA's with the mean of the column:
df_train_test_dummies = df_train_test_dummies.fillna(df_train_test_dummies.mean())

df_train_test_dummies = pd.get_dummies(df_train_test_dummies, columns=['MoSold'])

#train test splitting
X = df_train_test_dummies[:TRAIN_SIZE]
X_test = df_train_test_dummies[TRAIN_SIZE:]
y = df_train.SalePrice

print('shape for XGBoost: ', X.shape)
#rmse 0.11683
model_xgb = XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

print(rmse_cv(model_xgb).mean())

model_xgb.fit(X, y)
y_xgb = model_xgb.predict(X_test)
y_xgb = np.expm1(y_xgb)
#LilghtGBM rmse 0.11582
model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=4,
                              learning_rate=0.05, 
                              n_estimators=1000,
                              max_bin=75, 
                              bagging_fraction=0.8,
                              bagging_freq=5, 
                              feature_fraction=0.2319,
                              feature_fraction_seed=9, 
                              bagging_seed=9,
                              min_data_in_leaf=6, 
                              min_sum_hessian_in_leaf=11)
print(rmse_cv(model_lgb).mean())

model_lgb.fit(X, y)
y_lgb = model_lgb.predict(X_test)
y_lgb = np.expm1(y_lgb)
#The pairplot for predeictions of LGBM and Lasso
predictions = pd.DataFrame({"lgb":y_lgb, "lasso":y_lasso})
predictions.plot(x = "lgb", y = "lasso", kind = "scatter")
#The pairplot for predeictions of LGBM and XGB
predictions = pd.DataFrame({"xgb":y_xgb, "lgb":y_lgb})
predictions.plot(x = "xgb", y = "lgb", kind = "scatter")
#mixing version of several classifiers
class MeanRegressor(BaseEstimator, RegressorMixin):  

    def __init__(self, regressor1=None, regressor2=None, regressor3=None, r1=0.33, r2=0.33):
        
        self.regressor1 = regressor1
        self.regressor2 = regressor2
        self.regressor3 = regressor3
        self.r1 = r1
        self.r2 = r2
        

    def fit(self, X, y=None):
       
        self.regressor1.fit(X, y)
        self.regressor2.fit(X, y)
        self.regressor3.fit(X, y)
        
        return self

    def predict(self, X, y=None):
        return self.r1 * self.regressor1.predict(X) + self.r2 * self.regressor2.predict(X) + (1 - self.r1 - self.r2)*self.regressor3.predict(X)
#0.10843
mr = MeanRegressor(model_lasso, model_lgb, model_xgb, 0.55, 0.39)
rmse_cv(mr)
#0.5 0.4 0.1 - best, because xgb make worse
preds = 0.55*y_lasso + 0.39*y_lgb + 0.06*y_xgb
preds = preds
solution = pd.DataFrame({"id":df_test.Id, "SalePrice":preds})

solution.to_csv("solution.csv", index = False)