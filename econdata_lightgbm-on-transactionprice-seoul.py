# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import lightgbm as lgb

from sklearn.linear_model import Ridge

import time

from sklearn import preprocessing

import warnings

import datetime

warnings.filterwarnings("ignore")

import gc

from tqdm import tqdm



from scipy.stats import describe

%matplotlib inline



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

import xgboost as xgb



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics

import lightgbm as lgb



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999

# Any results you write to the current directory are saved as output.



# Any results you write to the current directory are saved as output.
df_train=pd.read_csv("../input/trainPrice.csv")
df_train.head()
df_train.info()
sch=pd.read_csv("../input/Schools.csv")
sch.info()
df_test=pd.read_csv("../input/testPrice.csv")
df_test.info()
df_test.head()
subwy=pd.read_csv("../input/Subways.csv")
subwy.head()
subwy.info()
subm=pd.read_csv("../input/submissionPrice.csv")
subm.head()
print(df_train.shape,df_test.shape,subm.shape,subwy.shape,sch.shape)
df_train.transaction_real_price.describe()
plt.figure(figsize=(12, 5))

plt.hist(df_train.transaction_real_price, bins=200)

plt.title('Histogram target counts')

plt.xlabel('Count')

plt.ylabel('transaction_real_price')

plt.show()
sns.set_style("whitegrid")

ax = sns.violinplot(x=df_train.transaction_real_price)

plt.show()
df_train.tail()
!ls ../input/
from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected=True)
miss_per = {}

for k, v in dict(df_train.isna().sum(axis=0)).items():

    if v == 0:

        continue

    miss_per[k] = 100 * float(v) / len(df_train)

    

import operator 

sorted_x = sorted(miss_per.items(), key=operator.itemgetter(1), reverse=True)

print ("There are " + str(len(miss_per)) + " columns with missing values")



kys = [_[0] for _ in sorted_x][::-1]

vls = [_[1] for _ in sorted_x][::-1]

trace1 = go.Bar(y = kys, orientation="h" , x = vls, marker=dict(color="#d6a5ff"))

layout = go.Layout(title="Missing Values Percentage", 

                   xaxis=dict(title="Missing Percentage"), 

                   height=400, margin=dict(l=300, r=300))

figure = go.Figure(data = [trace1], layout = layout)

iplot(figure)
df_train.isnull().values.any()
df_test.isnull().values.any()
# some out of range int is a good choice

df_train.fillna(-999, inplace=True)

df_test.fillna(-999, inplace=True)
df_train.head()
df_train.dtypes

df_test.dtypes
df_train["heat_type"].value_counts()
df_train["heat_type"] = df_train["heat_type"].astype('object')

df_train.dtypes
df_train=pd.get_dummies(df_train, columns=["heat_type"])

df_test=pd.get_dummies(df_test, columns=["heat_type"])

df_train=pd.get_dummies(df_train, columns=["front_door_structure"])
df_test=pd.get_dummies(df_test, columns=["front_door_structure"])
df_train=pd.get_dummies(df_train, columns=["heat_fuel"])
df_test=pd.get_dummies(df_test, columns=["heat_fuel"])
df_train.head()
df_train['elapsed_time'] = ([2019]- df_train['year_of_completion'])

df_test['elapsed_time'] = ([2019] - df_test['year_of_completion'])

df_train.head()
df_train=df_train.drop(['key', 'city','transaction_year_month','transaction_date','year_of_completion','address_by_law','room_id'],axis=1)
df_train=df_train.drop(['apartment_id'],axis=1)
df_train.head()
df_test=df_test.drop(['city','transaction_year_month','transaction_date','year_of_completion','address_by_law','room_id'],axis=1)
df_test=df_test.drop(['apartment_id'],axis=1)
df_test.head()
df_train.shape,df_train.size,df_test.shape,df_test.size
df_test.tail()
train_X=df_train.drop(['transaction_real_price'],axis=1)
test_X=df_test.drop(['transaction_real_price','key'],axis=1)
train_y=df_train['transaction_real_price']
def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 30,

        "min_child_weight" : 50,

        "learning_rate" : 0.05,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.7,

        "bagging_frequency" : 5,

        "bagging_seed" : 2018,

        "verbosity" : -1

    }

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    evals_result = {}

    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100, evals_result=evals_result)

    

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result







pred_test = 0

kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)

for dev_index, val_index in kf.split(df_train):

    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

    pred_test += pred_test_tmp

pred_test /= 5.
fig, ax = plt.subplots(figsize=(12,10))

lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()
sub = pd.DataFrame()

sub['key'] = df_test['key']

sub['transaction_real_price']= pred_test

sub.to_csv('submission.csv',index=False)