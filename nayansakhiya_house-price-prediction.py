# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

import lightgbm as lgb
# let's import training and testing data



train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data.head()
test_data.head()
#check the numbers of samples and features



print("The train data  size  is : {} ".format(train_data.shape))

print("The test data size is : {} ".format(test_data.shape))
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column



train_data["SalePrice"] = np.log1p(train_data["SalePrice"])
# features engineering



y = train_data.SalePrice.values

all_data = pd.concat((train_data.loc[:,'MSSubClass':'SaleCondition'],

                      test_data.loc[:,'MSSubClass':'SaleCondition']))

print("all_data size is : {}".format(all_data.shape))
# let's handle missing values in all data



total_of_all = all_data.isnull().sum().sort_values(ascending=False)

percent_of_all = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_of_all, percent_of_all], axis=1, keys=['Total', 'Percent'])

missing_data_test.head(20)
# let's impute missing values



# data description says NA means "No Pool"

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")



# data description says NA means "no misc feature"

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")



# data description says NA means "no alley access"

all_data["Alley"] = all_data["Alley"].fillna("None")



# data description says NA means "no fence"

all_data["Fence"] = all_data["Fence"].fillna("None")



# data description says NA means "no fireplace"

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")



all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



# replace with None

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'): 

    all_data[col] = all_data[col].fillna('None')



# replace with 0

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

    

# NaN means that there is no basement

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')



# zero for having no basement

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)



# We can fill 0 for the area and None for the type

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)



all_data = all_data.drop(['MSZoning'], axis=1)

all_data = all_data.drop(['Utilities'], axis=1)



# data description says NA means typical

all_data["Functional"] = all_data["Functional"].fillna("Typ")



all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
from scipy.stats import skew



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
# get dummy data



all_data = pd.get_dummies(all_data)

print(all_data.shape)
X_train = all_data[:train_data.shape[0]]

X_test = all_data[train_data.shape[0]:]
# Validation function



n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

model_lasso.fit(X_train, y)
model_enet = ElasticNetCV(alphas = [1, 0.1, 0.001, 0.0005], l1_ratio=.9, random_state=3)

model_enet.fit(X_train, y)
model_ridge = RidgeCV(alphas=[0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75])

model_ridge.fit(X_train, y)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, random_state =7)

model_xgb.fit(X_train,y)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(X_train,y)
score = rmsle_cv(model_lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_enet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_ridge)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
# avarage based model



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

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
averaged_models = AveragingModels(models = (model_lasso, model_xgb, model_lgb))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# define evalution function



def rmsle_pred(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
# train our model and predict value



averaged_models.fit(X_train, y)

averaged_train_pred = averaged_models.predict(X_train)

averaged_pred = np.expm1(averaged_models.predict(X_test))

print(rmsle_pred(y, averaged_train_pred))
# submit predicted values to submission file



submission = pd.DataFrame()

submission['Id'] = test_data.Id

submission['SalePrice'] = averaged_pred

submission.to_csv('submission.csv',index=False)