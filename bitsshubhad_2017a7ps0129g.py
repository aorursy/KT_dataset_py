import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np

import sklearn.linear_model

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split

import sklearn.decomposition

import sklearn.ensemble

import warnings

warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt

import sklearn.feature_selection

import sklearn.metrics

from sklearn.externals import joblib

from scipy.stats import skew

import sklearn.model_selection

import sklearn.kernel_ridge

from scipy.special import boxcox1p

import sklearn.pipeline

import sklearn.svm

from mlxtend.regressor import StackingCVRegressor
data = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv').drop('id', axis = 1)
def remove_highly_correlated(df):

    not_in_test = [('b28', 'b56'),

                  ('b34', 'b41'),

                  ('b41', 'b34'),

                  ('b42', 'b67'),

                  ('b44', 'b54'),

                  ('b54', 'b44'),

                  ('b56', 'b28'),

                  ('b67', 'b42')]

    ncdf = df.copy()

    df_corr = df.corr()

    removed = []

    for row in df_corr.keys():

        if row in ncdf.columns:

            for col in df_corr[row].keys():

                if col == row:

                    continue

                if col in ncdf.columns and df_corr[row][col]>=0.99 and (row, col) not in not_in_test:

#                     print(row, col, df_corr[row][col])

                    removed += [col]

                    ncdf = ncdf.drop(col, axis = 1)

    print('removed: ', removed)

    return ncdf
def fix_skew(df):

    bcdf = df.copy()

    skewed_feats = bcdf.apply(lambda x: skew(x)).sort_values(ascending=False)

    skewness = pd.DataFrame({'Skew' :skewed_feats})

    skewness = skewness[abs(skewness) >= 2.7].dropna()

    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

    skewed_features = skewness.index

    lam = 0.15

    for feat in skewed_features:

        bcdf[feat] = boxcox1p(bcdf[feat], lam)

    print(skewed_features)

    return bcdf
def make_categorical(df):

    catdf = df.copy()

    categorical = [col for col in catdf.columns if not col.startswith('a') and len(catdf[col].unique())<=2]

    print('features that are converted to categorical: ', categorical)

    catdf = pd.get_dummies(data=catdf,columns=categorical)

    categorical = [col for col in catdf.columns if len(catdf[col].unique())<=2]

    return catdf, categorical
df = data.copy()



const = [col for col in df if df[col].std() == 0]

df = df.drop(const, axis = 1)

print('constant: ', const)



T = 1439

df.insert(0, "time phase", df['time']/T - np.floor(df['time']/T))



df = fix_skew(df)

df = remove_highly_correlated(df)

df, categorical_features = make_categorical(df)

print('categorical features: ', categorical_features)
features = [col for col in df if col != 'label'] 

numerical_features = [col for col in features if col not in categorical_features]

X = df[features]

y = df['label']

print(features), len(features)
kf = sklearn.model_selection.KFold(n_splits=12, random_state=42, shuffle=True)

ridge_alphas = [0, 1e-4, 1e-2, 0.1]

ridge = sklearn.pipeline.make_pipeline(RobustScaler(), sklearn.linear_model.RidgeCV(alphas=ridge_alphas, cv=kf))

svr = sklearn.pipeline.make_pipeline(RobustScaler(), sklearn.ensemble.BaggingRegressor(sklearn.svm.LinearSVR(),

                                                                                       max_samples = 0.4,

                                                                                      verbose = 1, n_jobs = -1))

gbr = sklearn.pipeline.make_pipeline(RobustScaler(), sklearn.ensemble.GradientBoostingRegressor(n_estimators=6000,

                                learning_rate=0.05,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42))

nubgbr = sklearn.ensemble.GradientBoostingRegressor(n_estimators=3000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)

stack_gen = StackingCVRegressor(regressors=(ridge, nubgbr, svr),

                                meta_regressor=ridge,

                                use_features_in_secondary=True)
print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))
print('GradientBoosting')

gbr_model_full_data = gbr.fit(X, y)
g = gbr_model_full_data.predict(X)

st = stack_gen_model.predict(np.array(X))
def blended_predictions(X, w):

    return w*g + (1-w)*st
pred = [sklearn.metrics.mean_squared_error(y, blended_predictions(X, w), squared = False) for w in np.arange(0, 1, 0.001)]
idx = np.argmin(pred)

w = (idx)*0.001

w, np.min(pred)
data = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')
df = data.copy()

constant = ['b10', 'b12', 'b26', 'b61', 'b81']



removed = ['b16', 'b85', 'b87', 'b29', 'b62', 'b76', 'b92', 'a0', 'b59', 'b75', 'b89', 'a3', 'b67', 'b88']

skewed_features = ['b58', 'b68', 'b72', 'b64', 'b21', 'b86', 'b82', 'b46', 'b40', 'b13',

       'b39', 'b36', 'b69', 'b60', 'b79', 'b24', 'b78', 'b0', 'b87', 'b19',

       'b48', 'b65', 'b45', 'b38', 'b50', 'b6', 'b18', 'b49', 'b71', 'b88',

       'b80', 'b2', 'b90', 'b16', 'b63', 'b17', 'b1', 'b5', 'b44', 'b54', 'b9',

       'b73', 'b83']

to_be_categorical_features = ['b28', 'b41']

categorical_features = ['a1', 'a2', 'a4', 'a5', 'a6', 'b28_0.0', 'b28_58.0', 'b41_0.0', 'b41_8.0']



df = df.drop(constant, axis = 1)



T = 1439

df.insert(0, "time phase", df['time']/T - np.floor(df['time']/T))





lam = 0.15

for feat in skewed_features:

    df[feat] = boxcox1p(df[feat], lam)

df = df.drop(removed, axis = 1)

df[to_be_categorical_features] = df[to_be_categorical_features].astype('float')

df = pd.get_dummies(data=df,columns=to_be_categorical_features)
features = ['time phase', 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b11', 'b13', 'b14', 'b15', 'b17', 'b18', 'b19', 'b20', 'b21', 'b22', 'b23', 'b24', 'b25', 'b27', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b40', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b50', 'b51', 'b52', 'b53', 'b54', 'b55', 'b56', 'b57', 'b58', 'b60', 'b63', 'b64', 'b65', 'b66', 'b68', 'b69', 'b70', 'b71', 'b72', 'b73', 'b74', 'b77', 'b78', 'b79', 'b80', 'b82', 'b83', 'b84', 'b86', 'b90', 'b91', 'b93', 'time', 'a1', 'a2', 'a4', 'a5', 'a6', 'b28_0.0', 'b28_58.0', 'b41_0.0', 'b41_8.0']

numerical_features = [col for col in features if col not in categorical_features]

X = df[features]

print(numerical_features)
y_hat = w*gbr_model_full_data.predict(X) + (1-w)*stack_gen_model.predict(X) 
outdf = pd.DataFrame({'id': data['id'], 'label': y_hat})

outdf.head()
import math

def find_nearest(array,value):

    idx = np.searchsorted(array, value, side="left")

    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):

        return array[idx-1]

    else:

        return array[idx]
outdf['label'] = [find_nearest(uniq, lab) for lab in outdf['label']]

outdf.head()
outdf.to_csv('out.csv', index = False)