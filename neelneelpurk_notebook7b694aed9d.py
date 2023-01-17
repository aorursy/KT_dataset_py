# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn import model_selection, preprocessing

import xgboost as xgb





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df_mr = pd.read_csv("../input/train.csv")

df_mr.head()
df_mr.shape
missing_df = df_mr.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df
shape = df_mr.shape

print("No of rows in this data :" + str(shape[0]))

print("No of unique value of IDs in this data :" + str(len(np.unique(df_mr['ID'])) ))
df_X = df_mr.drop(["ID","y"], axis=1)

print('Feature types:')

df_X.dtypes.value_counts()
counts = [[], [], []]

cols = df_X.columns

for c in cols:

    typ = df_X[c].dtype

    uniq = len(np.unique(df_X[c]))

    if uniq == 1: counts[0].append(c)

    elif uniq == 2 and typ == np.int64: counts[1].append(c)

    else: counts[2].append(c)



print('Constant features: {} Binary features: {} Categorical features: {}\n'.format(*[len(c) for c in counts]))



print('Constant features:', counts[0])

print('Categorical features:', counts[2])
df = df_mr.copy()

for f in df.columns:

    if df[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[f].values)) 

        df[f] = lbl.transform(list(df[f].values))

df.head()
train_y = df.y.values

train_X = df.drop(["y",'ID'], axis=1)



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse', 

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=30, height=0.8, ax=ax)

plt.show()


arr = [5,10,8,6,1,2,3]

for i in arr :

    var_name = "X" + str(i)

    col_order = np.sort(df_mr[var_name].unique()).tolist()

    plt.figure(figsize=(12,6))

    sns.countplot(x=var_name, data=df_mr)

    plt.xlabel(var_name, fontsize=12)

    plt.ylabel('Occurance', fontsize=12)

    plt.title("Occurance of"+var_name, fontsize=15)

    plt.show()

    plt.figure(figsize=(12,6))

    sns.boxplot(x=var_name, y='y', data=df_mr)

    plt.xlabel(var_name, fontsize=12)

    plt.ylabel('y', fontsize=12)

    plt.title("Distribution of y variable with "+var_name, fontsize=15)

    plt.show()