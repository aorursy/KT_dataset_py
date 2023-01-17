# Basic

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Other

#import scipy.stats as stats

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



#Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



# Model

from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score



#System

import os

import warnings



#Configure Defaults

warnings.filterwarnings('ignore')

%matplotlib inline

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

#pd.set_option('display.max_columns', 500)
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train.shape
test = pd.read_csv('../input/test.csv')

test.shape
desc = open('../input/data_description.txt', 'r')

print(desc.read())
train.head()
train.columns
# Target variable

sns.distplot(train['SalePrice'], fit=stats.norm)
# Correct the skew

train.SalePrice = np.log1p(train.SalePrice)
# Check target distribution

sns.distplot(train['SalePrice'], fit=stats.norm)
# Save test ID

test_ID = test['Id']



# Save train target

target = train.SalePrice.values



# Get split marker

split = len(train)



# Concate data

data = pd.concat((train, test)).reset_index(drop=True)



# Drop index

data.drop("Id", axis=1, inplace=True)



# Drop target column

data.drop("SalePrice", axis=1, inplace=True)



# Check

data.shape
plt.scatter(train.GrLivArea, train.SalePrice)
train = train[train.GrLivArea < 4000]

plt.scatter(train.GrLivArea, train.SalePrice)
sns.heatmap(data.isnull())
na = data.isnull().sum() / len(data) * 100

na.sort_values(ascending=False).head(10)
features = data



# Some of the non-numeric predictors are stored as numbers; we convert them into strings 

features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)



features['Functional'] = features['Functional'].fillna('Typ')

features['Electrical'] = features['Electrical'].fillna("SBrkr")

features['KitchenQual'] = features['KitchenQual'].fillna("TA")

features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])

features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])

features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])



features["PoolQC"] = features["PoolQC"].fillna("None")



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    features[col] = features[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    features[col] = features[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    features[col] = features[col].fillna('None')



features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



objects = []

for i in features.columns:

    if features[i].dtype == object:

        objects.append(i)



features.update(features[objects].fillna('None'))



features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



# Filling in the rest of the NA's



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics.append(i)

features.update(features[numerics].fillna(0))



numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in features.columns:

    if features[i].dtype in numeric_dtypes:

        numerics2.append(i)



skew_features = features[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    features[i] = boxcox1p(features[i], boxcox_normmax(features[i] + 1))



features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)



features['YrBltAndRemod']=features['YearBuilt']+features['YearRemodAdd']

features['TotalSF']=features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']



features['Total_sqr_footage'] = (features['BsmtFinSF1'] + features['BsmtFinSF2'] +

                                 features['1stFlrSF'] + features['2ndFlrSF'])



features['Total_Bathrooms'] = (features['FullBath'] + (0.5 * features['HalfBath']) +

                               features['BsmtFullBath'] + (0.5 * features['BsmtHalfBath']))



features['Total_porch_sf'] = (features['OpenPorchSF'] + features['3SsnPorch'] +

                              features['EnclosedPorch'] + features['ScreenPorch'] +

                              features['WoodDeckSF'])



# simplified features

features['haspool'] = features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

features['has2ndfloor'] = features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasgarage'] = features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

features['hasbsmt'] = features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

features['hasfireplace'] = features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)



data = features
'''# Drop mostly empty features

data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence'], axis = 1)'''
'''# Set None = NAN for the feature

data['FireplaceQu'].replace('None', np.nan, inplace=True)



# Create boolean flag for roughly 50% NAs

data["FireplaceQu_Flag"] = data["FireplaceQu"].notnull().astype('int')



# Drop original

data = data.drop(['FireplaceQu'], axis = 1)'''
'''# Fill median into ~1/6 NAs

data.LotFrontage = data.LotFrontage.fillna(data.LotFrontage.median())'''
'''# Select columns due to theirs data type

float_col = data.select_dtypes('float')

int_col = data.select_dtypes('int')

object_col = data.select_dtypes('object')'''
'''# Remove and impute numerical features

for f in float_col: data[f] = data[f].fillna(data[f].median())

   #if data[f].isnull().sum() / data.shape[0] > 0.1667: del data[f] # Remove 1/6+ of NANs

   #else: data[f] = data[f].fillna(data[f].mean()) # Impute others with a mean value'''
'''# Remove and impute numerical features

for i in int_col: data[i] = data[i].fillna(data[i].mode()[0])

   #if data[i].isnull().sum() / data.shape[0] > 0.1667: del data[f] # Remove 1/6+ of NANs

   #else: data[i] = data[i].fillna(data[i].mode()[0]) # Impute others with a mean value'''
'''for o in object_col: data[o] = data[o].fillna('Unknown')'''
'''# These are actually categorical

num_cat = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']

data[num_cat] = data[num_cat].astype(str)'''
'''# Create new features

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']



data['Total_sqr_footage'] = (data['BsmtFinSF1'] + data['BsmtFinSF2'] +

                                 data['1stFlrSF'] + data['2ndFlrSF'])



data['Total_Bathrooms'] = (data['FullBath'] + (0.5*data['HalfBath']) + 

                               data['BsmtFullBath'] + (0.5*data['BsmtHalfBath']))



data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +

                              data['EnclosedPorch'] + data['ScreenPorch'] +

                             data['WoodDeckSF'])'''
'''# Create boolean flags

data['Pool_Flag'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

data['2ndfloor_Flag'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

data['Garage_Flag'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

data['Bsmt_Flag'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data['Fireplace_Flag'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)'''
'''num_f = data.dtypes[data.dtypes != "object"].index



# Check the skew of all numerical features

skew_f = data[num_f].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skew = pd.DataFrame({'Skew' :skew_f})

skew.head(10)'''
'''from scipy.special import boxcox1p



skew = skew[abs(skew) > 0.5]

print("There are {} skewed numerical features to Box Cox transform".format(skew.shape[0]))



skew_f = skew.index

lam = 0.15

for f in skew_f:

    #all_data[feat] += 1

    data[f] = boxcox1p(data[f], lam)'''
data = pd.get_dummies(data, prefix_sep='_', drop_first=True) # Drop originall feature to avoid multi-collinearity
data.shape
# Remove high-variance features

'''from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

sel.fit_transform(data)''' # Array is the output
# TODO: feature selection
data.head()
# Split data

train = data[:split]

test = data[split:]



# Get train variables for a model

x = train

y = target



# Train data split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.22, random_state=101)
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



from mlxtend.regressor import StackingCVRegressor
#Validation function

n_folds = 5



def rmsle_cv(model, X, y):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
#svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
XGB = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
LGBM = lgb.LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )
stack = StackingCVRegressor(regressors=(KRR, lasso, ENet, XGB, LGBM),

                                meta_regressor=XGB,

                                use_features_in_secondary=True)
models = [KRR, lasso, ENet, XGB, LGBM]



for m in models:

    score = rmsle_cv(m, X_train, y_train)

    print("Model score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
models.append(stack)



for m in models:

    m.fit(X_train, y_train)
'''print('START Fit')



print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))'''
def assembly(X):

    result = 0

    for m in models:

        result += m.predict(X) 

    return result / len(models) # Avg predict value of all models
def assembly(X):

    return ((0.1 * KRR.predict(X)) + \

            (0.1 * lasso.predict(X)) + \

            (0.1 * ENet.predict(X)) + \

            (0.1 * XGB.predict(X)) + \

            (0.1 * LGBM.predict(X)) + \

            (0.5 * stack.predict(np.array(X))))
'''print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))'''
print('RMSLE score on train data:')

print(rmsle(y_train, assembly(X_train)))
pred = np.floor(np.expm1(assembly(test)))
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = pred

sub.to_csv('submission.csv',index=False)
sub.head()