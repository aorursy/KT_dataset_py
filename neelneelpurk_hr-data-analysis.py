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



# Any results you write to the current directory are saved as output.|
df_hr = pd.read_csv("../input/HR_comma_sep.csv")

df_hr.head()
df_hr.info()
df = df_hr.copy()

for f in df.columns:

    if df[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[f].values)) 

        df[f] = lbl.transform(list(df[f].values))

df.head()


        

train_y = df.left.values

train_X = df.drop(["left"], axis=1)



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'binary:logistic',

    'eval_metric': 'auc',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
df1 = df_hr[['satisfaction_level','last_evaluation', 'average_montly_hours','left']]

sns.pairplot(df1,hue='left',palette='Dark2') #Drawing the scatterplot
#Boxplot

var1 = 'left'

var2 = 'average_montly_hours'

g = sns.FacetGrid(df, col = var1)

g.map(sns.boxplot, var2)

np.mean(df[df[var1]==1][var2]),np.mean(df[df[var1]==0][var2])
#Boxplot

var1 = 'left'

var2 = 'time_spend_company'

g = sns.FacetGrid(df, col = var1)

g.map(sns.boxplot, var2)

np.mean(df[df[var1]==1][var2]),np.mean(df[df[var1]==0][var2])
#Boxplot

var1 = 'left'

var2 = 'number_project'

g = sns.FacetGrid(df, col = var1)

g.map(sns.boxplot, var2)

np.mean(df[df[var1]==1][var2]),np.mean(df[df[var1]==0][var2])
sns.factorplot("sales", col="left", col_wrap=4, data=df_hr, kind="count", size=10, aspect=.8)
train_y = df.salary.values

train_X = df.drop(["salary"], axis=1)



xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'multi:softmax',

    "num_class" : 3,

    'eval_metric': 'auc',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()