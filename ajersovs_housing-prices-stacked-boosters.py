import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test2=test.copy()

full = [train,test]

train.shape
pd.set_option('display.max_columns', 500)

train.head()

train_miss = train.isnull().sum(axis = 0).sort_values(ascending=False)

test_miss = test.isnull().sum(axis = 0).sort_values(ascending=False)
plt.figure(figsize=(15,4))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
nanvals = pd.concat([train_miss,test_miss], axis=1).sort_values(by=0, ascending=False)
train = train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1)

test = test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1)
train.head()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train_strings = train.select_dtypes(include ='object') 

train_numeric = train.drop(train_strings, axis=1)
nanvals = nanvals[(nanvals[0]!=0)|(nanvals[1]!=0)]
list(nanvals.index.unique())

cols_with_miss = ['LotFrontage','GarageFinish','GarageQual','GarageType','GarageYrBlt','GarageCond',\

 'BsmtExposure','BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1','MasVnrType','MasVnrArea']
strings = []

numerics = []

for col in cols_with_miss:

    if col in list(train_strings.columns):

        strings.append(col)

    else:

        numerics.append(col)
plt.figure(figsize=(15,4))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
for col in numerics:

    train[col] = train[col].fillna(train[col].mean())

for col in strings:

    train[col] = train[col].fillna(train[col].mode())
for col in numerics:

    test[col] = test[col].fillna(test[col].mean())

for col in strings:

    test[col] = test[col].fillna(test[col].mode())

    
train = train.fillna(method='ffill')

test = test.fillna(method='ffill')
plt.figure(figsize=(15,4))

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(15,12))

sns.heatmap(train_numeric.corr(),cmap='viridis')
full = [train, test]

for df in full:

    df['MSSubClass'] = df['MSSubClass'].apply(str)

    df['OverallCond'] = df['OverallCond'].astype(str)

    df['YrSold'] = df['YrSold'].astype(str)

    df['MoSold'] = df['MoSold'].astype(str)
numeric = list(train.drop(train_strings, axis=1).columns)

numeric = numeric[:-1]

list(train.select_dtypes(include ='object').columns)

objects = ['MSSubClass','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig',

           'LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallCond',

           'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond',

           'Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating',

           'HeatingQC','CentralAir','Electrical','KitchenQual','Functional','GarageType','GarageFinish',

           'GarageQual','GarageCond','PavedDrive','MoSold','YrSold','SaleType','SaleCondition'] 
train[objects].describe()
list_of_huge_cats = ['Neighborhood', 'Exterior1st', 'Exterior2nd']
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

full = [train, test]

for df in full:

    for col in objects:

        df[col] = label_encoder.fit_transform(df[col])
train.head()
for col in list_of_huge_cats:  

    train[col+'_cut'] =pd.cut(train[col], 4)
for col in list_of_huge_cats:

    print(train[col+'_cut'].value_counts())
full = [train, test]

for df in full:    

    df.loc[ df['Neighborhood'] <=6, 'Neighborhood'] = 0

    df.loc[(df['Neighborhood'] > 6) & (df['Neighborhood'] <= 12), 'Neighborhood'] = 1

    df.loc[(df['Neighborhood'] > 12) & (df['Neighborhood'] <= 18), 'Neighborhood'] = 2

    df.loc[ df['Neighborhood'] > 18, 'Neighborhood'] = 3

    

for df in full:    

    df.loc[ df['Exterior1st'] <=3.5, 'Exterior1st'] = 0

    df.loc[(df['Exterior1st'] > 3.5) & (df['Exterior1st'] <= 7), 'Exterior1st'] = 1

    df.loc[(df['Exterior1st'] > 7) & (df['Exterior1st'] <= 10.5), 'Exterior1st'] = 2

    df.loc[ df['Exterior1st'] > 10.5, 'Exterior1st'] = 3

    

for df in full:    

    df.loc[ df['Exterior2nd'] <=3.75, 'Exterior2nd'] = 0

    df.loc[(df['Exterior2nd'] > 3.75) & (df['Exterior2nd'] <= 7.5), 'Exterior2nd'] = 1

    df.loc[(df['Exterior2nd'] > 7.5) & (df['Exterior2nd'] <= 11.25), 'Exterior2nd'] = 2

    df.loc[ df['Exterior2nd'] > 11.25, 'Exterior2nd'] = 3
train = train.drop(train[['Neighborhood_cut','Exterior1st_cut','Exterior2nd_cut']], axis=1)
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']

test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
from sklearn.preprocessing import StandardScaler

X = train.drop('SalePrice', axis=1)

y = np.log1p(train["SalePrice"])

scaler = StandardScaler()

scaler.fit(train[numeric])

X[numeric] = scaler.transform(X[numeric])

test[numeric] = scaler.transform(test[numeric])
X.head()
X = X.drop('Id',axis=1)

test = test.drop('Id', axis=1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_log_error

import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.model_selection import KFold, cross_val_score

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



score = rmsle_cv(model_xgb)

print("\nXGboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = rmsle_cv(gboost)

print("\nGradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

score = rmsle_cv(model_lgb)

print("\nLight GBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

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
averaged_models = AveragingModels(models = (model_xgb, gboost, model_lgb))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# averaged_models.fit(X_train,y_train)

# preds = averaged_models.predict(X_test)

# preds_test = averaged_models.predict(test)

# print(mean_absolute_error(y_test,preds))



gboost.fit(X_train,y_train)

preds = gboost.predict(X_test)

preds_test = gboost.predict(test)

print(mean_absolute_error(y_test,preds))
submission = pd.DataFrame()

submission['Id'] = test2.Id

submission['SalePrice'] = np.expm1(preds_test)

submission.to_csv('submission.csv',index=False)