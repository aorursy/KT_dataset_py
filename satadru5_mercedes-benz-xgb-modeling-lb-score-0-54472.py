# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing

import xgboost as xgb

color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head(5)
train.shape
for f in train.columns:

    if train[f].dtype=='object':

        print(f)
train.isnull().sum().sort_values(ascending=False)
plt.figure(figsize=(8,9))

plt.plot(train['y'])
plt.figure(figsize =(6,7))

sns.regplot(x=train.index.values, y=np.sort(train.y.values),data=train)

plt.xlabel('index')

plt.ylabel
plt.hist(train['y'])
sns.distplot(train.y.values,bins=30,kde=False)
dtype_df = train.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
missing_df = train.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')

missing_df
var_name = "X0"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X0"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X1"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X2"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X3"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train, order=col_order,jitter=True)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X4"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.stripplot(x=var_name, y='y', data=train, order=col_order,jitter=True)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X3"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.violinplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X6"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.violinplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X6"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.boxplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
var_name = "X8"

col_order = np.sort(train[var_name].unique()).tolist()

plt.figure(figsize=(12,6))

sns.violinplot(x=var_name, y='y', data=train, order=col_order)

plt.xlabel(var_name, fontsize=12)

plt.ylabel('y', fontsize=12)

plt.title("Distribution of y variable with "+var_name, fontsize=15)

plt.show()
#convert Categorical to continious through hot encoding

for f in train.columns:

    if train[f].dtype=='object':

        print(f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values.astype('str')) + list(test[f].values.astype('str')))

        train[f] = lbl.transform(list(train[f].values.astype('str')))

        test[f] = lbl.transform(list(test[f].values.astype('str')))



train.head(3)
train=train.drop(['ID'],axis=1)
label=train['y']
train=train.drop(['y'],axis=1)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'min_child_weight':1,

    'silent': 1,

    'seed':0

}
dtrain = xgb.DMatrix(train, label)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
#xgtrain = xgb.DMatrix(train, label, feature_names=train.columns)

#xgtest = xgb.DMatrix(val_X, val_y, feature_names=val_X.columns)

#watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

#num_rounds = 100 # Increase the number of rounds while running in local

#model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=5)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
test.head(3)
ID=test['ID']

test=test.drop(['ID'],axis=1)
dtest=xgb.DMatrix(test)
pred=model.predict(dtest)
output= pd.DataFrame({'ID' : ID, 'y' : pred})
output.head(5)
output.to_csv('preds_XGB.csv', index=False)