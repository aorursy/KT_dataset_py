import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from sklearn.decomposition import PCA

%matplotlib inline



pal = sns.color_palette()



print('# File sizes')

for f in os.listdir('../input/'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize("../input/"+f) / 1000000, 2)) + 'MB')
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv("../input/test.csv")

print("Size of the train data is : {} {}".format(*df_train.shape))

print("Size of the test data is : {} {}".format(*df_test.shape))
#for testing purpose

test = pd.read_csv("../input/test.csv")
print(df_train.head())

counts = [[],[],[]]

for col in df_train.columns:

    if len(df_train[col].unique()) == 1:

        counts[0].append(col)

    elif len(df_train[col].unique()) == 2:

        counts[1].append(col)

    else:

        counts[2].append(col)
df_train.shape
rare = []

for cols in counts[1]:

    if df_train[cols].value_counts()[1] < 5:

        rare.append(cols)

df_train.dtypes.value_counts()
object_features = df_train.select_dtypes(include = ["O"])

for cols in object_features.columns:

    if cols in counts[1]:

        print("Column {} contains binary variables".format(cols))

    else:

        print("Column {} contains categorical variables".format(cols))

object_features.describe()
plt.scatter(range(df_train.shape[0]), np.sort(df_train.y.values))
mean_x0 = df_train[['X0', 'y']].groupby(['X0'], as_index=False).median()

mean_x0.columns = ['X0', 'mean_x0']



df_train = pd.merge(df_train, mean_x0, on='X0', how='outer')



mean_x1 = df_train[['X1', 'y']].groupby(['X1'], as_index=False).median()

mean_x1.columns = ['X1', 'mean_x1']



df_train = pd.merge(df_train, mean_x1, on='X1', how='outer')



mean_x2 = df_train[['X2', 'y']].groupby(['X2'], as_index=False).median()

mean_x2.columns = ['X2', 'mean_x2']



df_train = pd.merge(df_train, mean_x2, on='X2', how='outer')



mean_x3 = df_train[['X3', 'y']].groupby(['X3'], as_index=False).median()

mean_x3.columns = ['X3', 'mean_x3']



df_train = pd.merge(df_train, mean_x3, on='X3', how='outer')



mean_x4 = df_train[['X4', 'y']].groupby(['X4'], as_index=False).median()

mean_x4.columns = ['X4', 'mean_x4']



df_train = pd.merge(df_train, mean_x4, on='X4', how='outer')



mean_x5 = df_train[['X5', 'y']].groupby(['X5'], as_index=False).median()

mean_x5.columns = ['X5', 'mean_x5']



df_train = pd.merge(df_train, mean_x5, on='X5', how='outer')



mean_x6 = df_train[['X6', 'y']].groupby(['X6'], as_index=False).median()

mean_x6.columns = ['X6', 'mean_x6']



df_train = pd.merge(df_train, mean_x6, on='X6', how='outer')



mean_x8 = df_train[['X8', 'y']].groupby(['X8'], as_index=False).median()

mean_x8.columns = ['X8', 'mean_x8']



df_train = pd.merge(df_train, mean_x8, on='X8', how='outer')



df_train = df_train.drop(['X0','X1','X2','X3','X4','X5','X6','X8'], axis=1).copy()
test = pd.merge(test, mean_x0, on='X0', how='left')

test['mean_x0'].fillna(test['mean_x0'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x1, on='X1', how='left')

test['mean_x1'].fillna(test['mean_x1'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x2, on='X2', how='left')

test['mean_x2'].fillna(test['mean_x2'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x3, on='X3', how='left')

test['mean_x3'].fillna(test['mean_x3'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x4, on='X4', how='left')

test['mean_x4'].fillna(test['mean_x4'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x5, on='X5', how='left')

test['mean_x5'].fillna(test['mean_x5'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x6, on='X6', how='left')

test['mean_x6'].fillna(test['mean_x6'].dropna().median(), inplace=True)



test = pd.merge(test, mean_x8, on='X8', how='left')

test['mean_x8'].fillna(test['mean_x8'].dropna().median(), inplace=True)



#test = test.drop(['ID'], axis=1).copy()

test_ = test.drop(['X0','X1','X2','X3','X4','X5','X6','X8'], axis=1).copy()
from sklearn.decomposition import PCA, FastICA

n_comp = 10



# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(df_train.drop(["y"], axis=1))

pca2_results_test = pca.transform(test_)



# ICA

ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_train = ica.fit_transform(df_train.drop(["y"], axis=1))

ica2_results_test = ica.transform(test_)



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    df_train['pca_' + str(i)] = pca2_results_train[:,i-1]

    test_['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    df_train['ica_' + str(i)] = ica2_results_train[:,i-1]

    test_['ica_' + str(i)] = ica2_results_test[:, i-1]

    

y_train = df_train["y"]

y_mean = np.mean(y_train)
()# mmm, xgboost, loved by everyone ^-^

import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 500, 

    'eta': 0.005,

    'max_depth': 4,

    'subsample': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}



# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(df_train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test_)



# xgboost, cross-validation

cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=700, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=50, 

                   show_stdv=False

                  )



num_boost_rounds = len(cv_result)

print(num_boost_rounds)



# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
# check f2-score (to get higher score - increase num_boost_round in previous cell)

from sklearn.metrics import r2_score



# now fixed, correct calculation

print(r2_score(dtrain.get_label(), model.predict(dtrain)))
# make predictions and save results

y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('submission1.csv'.format(xgb_params['max_depth']), index=False)
x_train = df_train.drop("y",axis = 1)

y_train = df_train["y"]
x_test = test_

x_test.head()
import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(x_test)



params = {}

params['objective'] = 'reg:linear'

params['eta'] = 0.02

params['max_depth'] = 4



def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=50, feval=xgb_r2_score, maximize=True, verbose_eval=10)
test_id = test["ID"]

p_test = clf.predict(d_test)



sub = pd.DataFrame()

sub['ID'] = test_id

sub['y'] = p_test

sub.to_csv('submission.csv', index=False)