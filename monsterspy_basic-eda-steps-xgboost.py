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
print(df_train.head())

counts = [[],[],[]]

for col in df_train.columns:

    if len(df_train[col].unique()) == 1:

        counts[0].append(col)

    elif len(df_train[col].unique()) == 2:

        counts[1].append(col)

    else:

        counts[2].append(col)
df_train.shape,df_train.shape
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
object_features_t = df_test.select_dtypes(include = ["O"])

for cols in object_features_t.columns:

    if cols in counts[1]:

        print("Column {} contains binary variables".format(cols))

    else:

        print("Column {} contains categorical variables".format(cols))

object_features_t.describe()
plt.scatter(range(df_train.shape[0]), np.sort(df_train.y.values))
df_train_ = df_train.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8','y'], axis=1)

df_test_ = df_test.drop(['ID','X0','X1','X2','X3','X4','X5','X6','X8'], axis=1)
dummy = pd.get_dummies(object_features)

dummy_t = pd.get_dummies(object_features_t)

dummy.shape,dummy_t.shape
from sklearn.decomposition import PCA

n_comp = 150



# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(dummy)

pca2_results_test = pca.fit_transform(dummy_t)





# Append decomposition components to datasets

for i in range(1, n_comp+1):

    df_train_['pca_' + str(i)] = pca2_results_train[:,i-1]

    df_test_['pca_' + str(i)] = pca2_results_test[:, i-1]
# PCA

pca = PCA(iterated_power='auto', n_components=220, random_state=None, svd_solver='auto')

pca2_results_train = pca.fit_transform(df_train_)

pca2_results_test = pca.fit_transform(df_test_)
x_train = pd.DataFrame(data = pca2_results_train[0:,0:])

x_test = pd.DataFrame(data = pca2_results_test[0:,0:])
import xgboost as xgb

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_train, df_train["y"], test_size=0.2, random_state=4242)



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
test_id = df_test["ID"]

p_test = clf.predict(d_test)



sub = pd.DataFrame()

sub['ID'] = test_id

sub['y'] = p_test

sub.to_csv('submission3.csv', index=False)