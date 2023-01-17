# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import preprocessing

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ", df.shape)

print("Test shape : ", test_df.shape)
df.head()
import seaborn as sns; sns.set(color_codes=True)



ulimit = df['y'].ix[df['y']>= 180]



plt.figure(figsize=(16,10))

sns.distplot(df.y.values, kde=True, rug=True)

plt.xlabel('Y prediction', fontsize=14)

plt.title('Distribution Graph', fontsize=20)

plt.show()
plt.figure(figsize=(16,10))

sns.distplot(np.log(df.y.values), kde=True, rug=True)

plt.xlabel('Y prediction', fontsize=14)

plt.title('Distribution Graph', fontsize=20)

plt.show()
type(df['X0'])
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder



lb = preprocessing.LabelBinarizer()

lb.fit_transform(["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"])
enc = OneHotEncoder(handle_unknown='ignore')

for col in df.columns:

    if col is ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:

        X = str(np.sort(df[col].unique()).tolist())

        enc.fit(X)

        enc.transform(df[col])
#Inspi

dict_ = {}

for col_1 in df.columns:

    if col_1 not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:

        valeur_unique = str(np.sort(df[col_1].unique()).tolist())

        

        tlist = dict_.get(valeur_unique, [])

        tlist.append(col_1)

        dict_[valeur_unique] = tlist[:]

        

for valeur_unique, colonnes in dict_.items():

    print("Les valeurs unique:", valeur_unique)

    print(colonnes)
var_name = 'X2'

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(16,10))

sns.boxplot(x=var_name, y='y', data=df, order=col_order)

plt.ylabel("Y")

plt.xlabel("X2")

plt.title('Distribution X2 with Y', fontsize=20)
var_name = "X3"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(16,10))

sns.violinplot(x=var_name, y='y', data=df, order=col_order)

plt.xlabel('X3', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of Y variable with X3", fontsize=20)

plt.show()
sns.set()

var_name = 'X5'

plt.figure(figsize=(16,10))

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)

sns.scatterplot(x= var_name, y='y', palette=cmap, sizes=(10, 200), data=df)

plt.xlabel('X5', fontsize=12)

plt.ylabel('Y', fontsize=12)

plt.title('Distribution X5 with Y', fontsize=20)
var_name = "X6"

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(16,10))

sns.violinplot(x=var_name, y='y', data=df, order=col_order)

plt.xlabel('X6', fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of Y variable with X6", fontsize=20)

plt.show()
var_name = 'X8'

col_order = np.sort(df[var_name].unique()).tolist()

plt.figure(figsize=(16,10))

sns.boxplot(x=var_name, y='y', data=df, order=col_order)

plt.ylabel("Y")

plt.xlabel("X8")

plt.title('Distribution Y with X8', fontsize=20)
var_name = "ID"

plt.figure(figsize=(16,10))

sns.regplot(x=var_name, y='y', data=df, scatter_kws={'alpha':0.5, 's':30})

plt.xlabel('ID', fontsize=12)

plt.ylabel('Y', fontsize=12)

plt.title("Distribution of Y variable with ID", fontsize=20)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_validate

from sklearn.metrics import r2_score
y_train = df['y'].values

y_mean = np.mean(y_train)

X_train = df.select_dtypes(int).drop(columns=['ID']).values

X_test = test_df.select_dtypes(int).drop(columns=['ID']).values
#cols_categorical = df.select_dtypes(object).columns

#for col in cols_categorical:

#    X = np.unique(np.concatenate((df[col].unique(), test_df[col].unique())))

#    enc = LabelEncoder()

#    enc.fit(X)

#    X_train = np.append(X_train, enc.transform(df[col]).reshape(-1, 1), axis=1)

#    X_test = np.append(X_test, enc.transform(test_df[col]).reshape(-1, 1), axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=4242)



X_train = xgb.DMatrix(X_train, label=y_train)

X_valid = xgb.DMatrix(X_valid, label=y_valid)

test = xgb.DMatrix(X_test)
XG_params = {

    'n_trees': 520, 

    'eta': 0.0045,

    'max_depth': 4,

    'subsample': 0.93,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean,

    'silent': 1

}
num_boost_rounds = 1250

# train model

rf = xgb.train(dict(XG_params, silent=0), X_train, num_boost_round=num_boost_rounds)

y_pred = rf.predict(test)

print(y_pred)
print(r2_score(X_train.get_label(), rf.predict(y_pred)))
y_pred = rf.predict(test)

y_test = pd.DataFrame({'ID': test_df['ID'].values, 'y': y_test})

y_test.to_csv('submissions.csv', index=False)

print(y_test)
#rf = RandomForestRegressor(n_estimators=10, random_state=42)

#rf = GradientBoostingRegressor(n_estimators=10, random_state=50, )
%%time

rf.fit(X_train, y_train)
res = cross_validate(rf, y_pred, scoring='r2', cv=4, n_jobs=-1, return_train_score=False)
res
rf.fit(X_train, y_train)

y_test = rf.predict(X_test)
#score = r2_score(X_test, y_pred)
y_test = pd.DataFrame({'ID': test_df['ID'].values, 'y': y_test})
y_test.to_csv('submissions.csv', index=False)

print(y_test)