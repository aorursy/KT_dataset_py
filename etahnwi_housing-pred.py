# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np

from sklearn import preprocessing

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

matplotlib.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print("training data size: {}".format(train.shape))

print("testing data shape: {}".format(test.shape))
train.columns
quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']
quantitative
qualitative
train[qualitative]
is_null = pd.DataFrame(train.isna().sum().reset_index())

test_is_null = pd.DataFrame(test.isna().sum().reset_index())

is_null_ = pd.concat([is_null,test_is_null], ignore_index=True,axis=1)

is_null_ = is_null_.rename(columns={1:'train_null',3:'test_null'})

is_null_ = is_null_.sort_values("train_null", ascending=False)

is_null_.head(30)
ntest = test.shape[0]

ntrain = train.shape[0]

y_train = train.SalePrice

all_data = pd.concat((train,test)).reset_index(drop=True)

all_data.drop(['SalePrice'],axis=1, inplace=True)

all_data.shape
all_data.isna().sum().sort_values(ascending=False)[:35]
all_data.KitchenQual.unique()
train = train[train.GrLivArea < 4500]

train.reset_index(drop=True, inplace=True)

train["SalePrice"] = np.log1p(train["SalePrice"])

y = train['SalePrice'].reset_index(drop=True)
all_data['PoolQC'] = all_data.PoolQC.fillna("None")

all_data['MiscFeature'] = all_data.MiscFeature.fillna("None")

all_data['Alley'] = all_data.Alley.fillna('None')

all_data['Fence'] = all_data.Fence.fillna("None")

all_data['FireplaceQu'] = all_data.FireplaceQu.fillna("None")

all_data['LotFrontage'] = all_data.groupby(['Neighborhood'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))



for col in ('GarageCond','GarageQual', 'GarageFinish','GarageType'):

    all_data[col] = all_data[col].fillna("None")

    

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)



all_data['GarageArea'] = all_data['GarageArea'].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

    

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')





all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

all_data['Functional'] = all_data['Functional'].fillna('Typ')

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data.SaleType.fillna(all_data.SaleType.mode()[0])

all_data['Electrical'] = all_data.Electrical.fillna(all_data.Electrical.mode()[0])



all_data = all_data.drop("Utilities",axis=1)
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data.KitchenQual.mode()[0])
all_data.isna().sum().sort_values(ascending=False)[:30]
qualitative
qualitative = qualitative.remove(Utilities)
#Label Encoder

from sklearn.preprocessing import LabelEncoder



label_all_data = all_data.copy()



label_encoder = LabelEncoder()

for col in qualitative:

    label_all_data[col] = label_encoder.fit_transform(all_data[col])
label_all_data
label_all_data['TotalSF'] = label_all_data['TotalBsmtSF']+label_all_data['1stFlrSF']+label_all_data['2ndFlrSF']
label_all_data.dtypes
#finding if features are skewed. 

num_feats = [f for f in label_all_data.columns if label_all_data.dtypes[f] != 'object']



skew_features = label_all_data[num_feats].skew().sort_values(ascending=False)

skew_features.head(30)

skew_feat = skew_features.index
# Lets transform using boxcox on this one.



from scipy.special import boxcox1p



for col in skew_feat:

    label_all_data[col] = boxcox1p(label_all_data[col],0.15)  

sns.kdeplot(label_all_data.TotalBsmtSF)
#So this will not work because i removed all the categorical data by doing Label Encoding on all CAT features.. This would be a good opportuniy

#to first see how our models perform when all CAT features have have been encoded and then re-running out models after only selecting certain features

# in out data to for encoding.



label_all_data = pd.get_dummies(label_all_data)

label_all_data.shape
X = label_all_data.iloc[:len(y), :]

X_sub = label_all_data.iloc[len(y):, :]

X.shape, y.shape, X_sub.shape

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
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
model = RandomForestRegressor()

score = cv_rmse(model)

print(score.mean())

print(score.std())
print("Random Forest")

forest_model = model.fit(X,y)
print('RMSLE score on train data')

preds = forest_model.predict(X)

print(rmsle(y,preds))
X_sub.head()
test_data_path = '../input/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

test_preds = rf_model_on_full_data.predict(test_X)





output = pd.DataFrame({'Id': X_sub.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)



test_preds = model.predict(X_sub)
test_preds = model.predict(X_sub)



output = pd.DataFrame({'Id': X_sub.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)