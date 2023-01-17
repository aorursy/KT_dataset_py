# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from datetime import datetime

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



import os
train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_df  = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print('='*20, 'the col of string', '='*20)

for col in train_df.columns:

    if train_df.dtypes[col] == 'object':

        print(col, train_df[col].unique().shape[0])

print('='*20, 'the col of number', '='*20)

for col in train_df.columns:

    if train_df.dtypes[col] != 'object':

        print(col, 'min:', train_df[col].min(), 'max:', train_df[col].max())
print('='*20, 'the count of na', '='*20)

nan_count = train_df.isna().sum()

nan_count = nan_count[nan_count > 0]

dict(nan_count.sort_values(ascending=False))
for col in train_df.columns:

    if train_df.dtypes[col] != 'object':

        plt.figure()

        plt.title(col)

        train_df[col].plot()
train_df.drop(['Id'], axis=1, inplace=True)

test_df.drop(['Id'], axis=1, inplace=True)
train_df.GrLivArea.describe()
train_df = train_df[train_df.GrLivArea < 4500]

train_df.reset_index(drop=True, inplace=True)
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

target = train_df['SalePrice']
train_features = train_df.drop(['SalePrice'], axis=1)

test_features = test_df

features = pd.concat([train_features, test_features]).reset_index(drop=True)
features.shape
plt.figure()

features['MSSubClass'].plot()

plt.figure()

features['YrSold'].plot()

plt.figure()

features['MoSold'].plot()
features['MSSubClass'] = features['MSSubClass'].apply(str)

features['YrSold'] = features['YrSold'].astype(str)

features['MoSold'] = features['MoSold'].astype(str)
print('='*20, 'the count of na', '='*20)

nan_count = features.isna().sum()

nan_count = nan_count[nan_count > 0]

nan_count.sort_values(ascending=False, inplace=True)

dict(nan_count)
for col in nan_count.keys():

    if features[col].dtype == 'object':

        features[col] = features[col].fillna("None")
for col in nan_count.keys():

    if features[col].dtype != 'object':

        print(col, features[col].min(), features[col].max())

        features[col] = features[col].fillna(features[col].mode()[0]) 
def encode(train_df, col):

    set_value = train_df[col].unique()

    encode = {key:value for key, value in zip(set_value, range(set_value.shape[0]))}

    train_df[col] = train_df[col].map(encode)



for col in features.columns:

    if features.dtypes[col] == 'object':

        encode(features, col)
features = features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
features.shape
X = features.iloc[:len(target), :]

X_sub = features.iloc[len(target):, :]

X.shape, target.shape, X_sub.shape
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
models = []

models.append(make_pipeline(RobustScaler(), RidgeCV(alphas=np.linspace(14.5, 15.5, 11).tolist(), cv=kfolds)))

models.append(make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=np.linspace(5e-5, 8e-4, 9).tolist(), random_state=42, cv=kfolds)))

models.append(make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=np.linspace(1e-4, 7e-4, 7).tolist(), cv=kfolds, l1_ratio=np.linspace(0.8, 1.0, 21).tolist()))  )

models.append(GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state = 42))

models.append(LGBMRegressor(objective='regression', 

                                       num_leaves=31,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=255, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.9,

                                       feature_fraction_seed=42,

                                       verbose=-1,

                                       ))

models.append(XGBRegressor(learning_rate=0.01,n_estimators=3460,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006))
import sys

sys.setrecursionlimit(1000000)

for model in models:

    print(model)

    model.fit(features.iloc[:len(target), :].to_numpy(), target)
Stacking = StackingCVRegressor(regressors=models,

                                meta_regressor=models[-2],

                                use_features_in_secondary=True)

Stacking.fit(features.iloc[:len(target), :].to_numpy(), target)

models.append(Stacking)
def predict(data):

    blending_weights = [0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.3]

    output = np.zeros(data.shape[0])

    for index, model in enumerate(models):

        output += blending_weights[index] * model.predict(data)

#         print(model.predict(data))

#     print(output)

    return output
test_data = features.iloc[len(target):, :].to_numpy()
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(predict(test_data)))
submission.head()
#sub_1 = pd.read_csv('../input/top-10-0-10943-stacking-mice-and-brutal-force/House_Prices_submit.csv')

#sub_2 = pd.read_csv('../input/hybrid-svm-benchmark-approach-0-11180-lb-top-2/hybrid_solution.csv')

#sub_3 = pd.read_csv('../input/lasso-model-for-regression-problem/lasso_sol22_Median.csv')

#submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(predict(test_data)))) + 

#                                (0.25 * sub_1.iloc[:,1]) + 

#                                (0.25 * sub_2.iloc[:,1]) + 

#                                (0.25 * sub_3.iloc[:,1]))
submission.to_csv("submission.csv", index=False)
submission.head()