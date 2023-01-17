print("Setup Complete")

import numpy as np

import pandas as pd

import os

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

import missingno as msno

import seaborn as sns
#Read Data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

print(X_full.shape)

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

print(X_full.describe())

#print(X_full.columns)
fig, ax = plt.subplots()

ax.scatter(x = X_full['GrLivArea'], y = X_full.SalePrice)

plt.show()
print(X_full['GrLivArea'].dtype)

X_full = X_full.drop(X_full[(X_full['GrLivArea'] > 4000) & (X_full['SalePrice'] < 300000)].index)

fig, ax = plt.subplots()

ax.scatter(x = X_full['GrLivArea'], y = X_full.SalePrice)

plt.show()
y = X_full.SalePrice

X_full.drop(['SalePrice'], axis = 1, inplace = True)
X_full["LotFrontage"] = X_full.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



input_full = pd.concat([X_full, X_test_full], axis = 0)



numeric_features = input_full._get_numeric_data()

print(numeric_features)



categoric_cols = list(set(X_full.columns) - set(numeric_features))

categoric_features = input_full[categoric_cols]

print(categoric_features)
from scipy import stats

from scipy.stats import norm, skew

sns.distplot(y, fit = norm);

fig = plt.figure

plt.show()
stats.probplot(y, plot=plt)
y = np.log(y)

sns.distplot(y, fit = norm);
stats.probplot(y, plot=plt)
numeric_features['MSSubClass']
for col in numeric_features:

    numeric_features[col] = numeric_features.fillna(0)



for col in ('Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'):

    categoric_features[col] = categoric_features.fillna("None")



for col in ('MasVnrType', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    categoric_features[col] = categoric_features[col].fillna('None')

    

for col in ('MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MSSubClass'):

    numeric_features[col] = numeric_features.fillna(0.0)

    

for col in ('MSZoning','Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType'):

    categoric_features[col] = categoric_features[col].fillna(categoric_features[col].mode()[0])

    

categoric_features["Functional"] = categoric_features["Functional"].fillna("Typ")



#Make sure every variable is cleaned up

features_w_na = pd.concat([numeric_features.isnull().sum(), categoric_features.isnull().sum()], axis = 0)

if(features_w_na[features_w_na > 0].empty):

    print("yay data is cleaned up!!!! Nice Job XD!")

else:

    print(features_w_na[features_w_na > 0])
categoric_features.drop(['Utilities'], axis=1)
skewed_cols = numeric_features.apply(lambda x: skew(x)).sort_values(ascending = False)

print(skewed_cols)
skewed_cols = skewed_cols[abs(skewed_cols) > 0.75]



from scipy.special import boxcox1p

skewed_features = skewed_cols.index

lam = 0.15

for feat in skewed_features:

    numeric_features[feat] = boxcox1p(X_full[feat], lam)
#checking skewness

skewed_cols = numeric_features.apply(lambda x: skew(x)).sort_values(ascending = False)

skewed_cols

#yay it looks slightly better!
print('Shape of input: {}'.format(X_full.shape))

print('Shape of test input: {}'.format(X_test_full.shape))

print(list(set(X_test_full.columns) - set(X_full.columns)))
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)



ohe_input_full = pd.DataFrame(ohe.fit_transform(categoric_features))

ohe_input_full.index = categoric_features.index
#seperate test and train from data

input_full = pd.concat([ohe_input_full, numeric_features], axis=1)



X_full = input_full.head(X_full.shape[0])

X_test_full = input_full.tail(X_test_full.shape[0])



print('Shape of input: {}'.format(X_full.shape))

print('Shape of test input: {}'.format(X_test_full.shape))

print(X_full.head)

print(X_test_full.head)
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype == "object"]



numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



numerical_transformer = SimpleImputer(strategy='constant')



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])





evi = [80]

score = []

for i in evi:

    model = RandomForestRegressor(n_estimators=i, random_state=0)

    clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])

    clf.fit(X_train, y_train)

    preds = clf.predict(X_valid)

    score.append(mean_absolute_error(y_valid, preds))

print('MAE:', score)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print(score)
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=1).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=2999, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lassoscore = rmsle_cv(lasso)
enetscore = rmsle_cv(ENet)
krrscore = rmsle_cv(KRR)
gboostscore = rmsle_cv(GBoost)
xgbscore = rmsle_cv(model_xgb)
lgbscore = rmsle_cv(model_lgb)
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

    

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   

        

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)