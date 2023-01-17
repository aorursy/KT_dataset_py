import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import ensemble, tree, linear_model

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.utils import shuffle



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
dir = "/kaggle/input/house-prices-advanced-regression-techniques/"

train = pd.read_csv(dir+'train.csv')

test = pd.read_csv(dir+'test.csv')

y = np.log1p(train["SalePrice"])

test_ID = test['Id']
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['Id'], axis=1, inplace=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)



for col in ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",

            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

           'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

           "MasVnrType", "MSSubClass"]:

    all_data[col] = all_data[col].fillna('None')



all_data["Functional"] = all_data["Functional"].fillna("Typ")

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars',

            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',

           "MasVnrArea"]:

    all_data[col] = all_data[col].fillna(0)

    

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))



for col in ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]:

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])



all_data = all_data.drop(['Utilities'], axis=1)



ntrain = train.shape[0]

ntest = test.shape[0]
#Check remaining missing values if any 

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

print("NA column:", len(all_data_na))
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# all_data['OverallCond'] = all_data['OverallCond'].astype(str)



all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
categorical_features = all_data.select_dtypes(include = ["object"]).columns

numerical_features = all_data.select_dtypes(exclude = ["object"]).columns

train_num = all_data[numerical_features]

train_cat = all_data[categorical_features]



print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))
# for c in categorical_features:

#     print(c, all_data[c].unique())

from sklearn import preprocessing

ordinals = ["MoSold", "YrSold"]

for c in ordinals:

    labels = all_data[c].unique()

    labels.sort()

    

    le = preprocessing.LabelEncoder()

    le.fit(labels)

    all_data[c] = le.transform(all_data[c]) 

    

qc = {

    "ExterQual": ["Ex", "Gd", "TA", "Fa", "Po"], 

    "ExterCond": ["Ex", "Gd", "TA", "Fa", "Po"], 

    "BsmtQual": ["Ex", "Gd", "TA", "Fa", "Po", "None"], 

    "BsmtCond": ["Ex", "Gd", "TA", "Fa", "Po", "None"],

    "BsmtExposure" : ["Gd", "Av", "Mn", "No", "None"],

    "BsmtFinType1": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"],

    "BsmtFinType2": ["GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf", "None"],

    "HeatingQC": ["Ex", "Gd", "TA", "Fa", "Po"], 

    "KitchenQual": ["Ex", "Gd", "TA", "Fa", "Po"], 

    "FireplaceQu": ["Ex", "Gd", "TA", "Fa", "Po", "None"],

    "GarageQual": ["Ex", "Gd", "TA", "Fa", "Po", "None"],

    "GarageCond": ["Ex", "Gd", "TA", "Fa", "Po", "None"],

    "PoolQC": ["Ex", "Gd", "TA", "Fa", "None"], 

}



for c in qc:

    labels = qc[c]

    

    le = preprocessing.LabelEncoder()

    le.fit(labels)

    all_data[c] = le.transform(all_data[c]) 

    

all_data.shape
# One-hot encoding for others

all_data = pd.get_dummies(all_data)

all_data.shape
train = all_data[:ntrain]

test = all_data[ntrain:]
# train = pd.concat([train_cat,train_num],axis=1)

# X_train,X_test,y_train,y_test = train_test_split(train,y,test_size = 0.3,random_state= 0)



# train.shape, X_train.shape, X_test.shape
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

def rmsle_cv(model, X, y):

    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

Xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

Lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
modelDesc = {

    "lasso": {"name": "Lasso", "obj": lasso},

    "ENet": {"name": "ENet", "obj": ENet},

    "KRR": {"name": "KRR", "obj": KRR},

    "GBoost": {"name": "GBoost", "obj": GBoost},

    "Xgb": {"name": "Xgb", "obj": Xgb},

    "Lgb": {"name": "Lgb", "obj": Lgb},

}
# Weighted Average model

WAmodel = {}

for m in modelDesc:

    obj = modelDesc[m]["obj"]

    obj.fit(train.values, y)

    

    scores = rmsle_cv(obj, train.values, y)

    print("Averaged score: {} {:.4f} ({:.4f})\n".format(modelDesc[m]["name"], scores.mean(), scores.std()))

    

    WAmodel[m] = (obj, scores.mean())
from sklearn.metrics import mean_squared_error



def weightAveragePredict(models, X):

    weights = [1.0/WAmodel[m][1] for m in WAmodel]

    weights = np.array(weights)

    weights = weights/weights.sum()

    

    yhat = None

    ind = 0

    for m in WAmodel:

        obj = WAmodel[m][0]

        p = obj.predict(X)

        p = weights[ind]*p

        ind += 1

        

        if yhat is None:

            yhat = p

        else:

            yhat = yhat + p

        

    return yhat

    

y_pred = weightAveragePredict(WAmodel, train.values)

score = np.sqrt(mean_squared_error(y, y_pred))

print(score)



from sklearn.linear_model import LinearRegression



def predictBaseModels(models, X):

    yhat = []

    ind = 0

    for m in WAmodel:

        obj = WAmodel[m][0]

        p = obj.predict(X)

        yhat.append(p)

    

    yhat = np.array(yhat)

    yhat = np.transpose(yhat)

    return yhat



def trainMetaModel(yhat, y):

    reg = LinearRegression().fit(yhat, y)

    return reg



yhat = predictBaseModels(WAmodel, train.values)

meta = trainMetaModel(yhat, y)



y_pred = meta.predict(yhat)

score = np.sqrt(mean_squared_error(y, y_pred))

print(score)
# y_pred = weightAveragePredict(WAmodel, test.values)



_y_pred = predictBaseModels(WAmodel, test.values)

y_pred = meta.predict(_y_pred)



exp_y_pred = np.expm1(y_pred)
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = exp_y_pred

sub.to_csv('submission.csv',index=False)