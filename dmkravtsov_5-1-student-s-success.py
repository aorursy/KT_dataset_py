# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 1000)

from sklearn.metrics import mean_absolute_error

import seaborn as sns
# in this kernel I'll work with student-por dataset due to better volume of samples

mat = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-mat.csv')

train = pd.read_csv('/kaggle/input/student-alcohol-consumption/student-por.csv')
train.info()
train['G'] = round((train['G1']+train['G2']+train['G3'])/3)

train = train.drop(['G1', "G2", "G3"], axis=1)


def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(train)
for col in train.drop(['G'], axis=1):

    train[col] = train[col].astype(str)

train = pd.get_dummies(train)


def basic_details(df):

    b = pd.DataFrame()

    b['Missing value'] = df.isnull().sum()

    b['N unique value'] = df.nunique()

    b['dtype'] = df.dtypes

    return b

basic_details(train)
sns.distplot(np.log1p(train['G']))
train.shape
#creating matrices for feature selection:

X = train.drop(['G'], axis=1)

y = train.G
#Correlation with output variable

cor = train.corr()

cor_target = (cor['G'])

#Selecting highly correlated features

relevant_features = cor_target

relevant_features.sort_values(ascending = False).head(1000)
import xgboost as xgb

from xgboost import cv

from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

def xgb_r2_score(preds, dtrain):

    labels = dtrain.get_label()

    return 'r2', r2_score(labels, preds)







params = {

        'objective':'reg:linear',

        'n_estimators': 1000,

        'booster':'gbtree',

        'max_depth':3,

        'eval_metric':'rmse',

        'learning_rate':0.1, 

        'min_child_weight':2,

        'subsample':0.5,

        'colsample_bytree':0.5,

        'seed':45,

        'reg_lambda':1,

        'reg_alpha':0.01,

        'gamma':0.45,

        'nthread':-1,

}





x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=29)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)





watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100,  feval=xgb_r2_score, maximize=True, verbose_eval=10)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(clf, max_num_features=50, height=0.8, ax=ax,color='r')

plt.show()