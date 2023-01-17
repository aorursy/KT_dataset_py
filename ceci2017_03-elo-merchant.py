import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

import seaborn as sns

from tqdm import tqdm

from datetime import datetime

import json

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split, KFold, cross_val_score 

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import LabelEncoder, Imputer

from sklearn import model_selection, preprocessing, metrics

import lightgbm as lgb



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from scipy.stats import skew 

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

# models

from xgboost import XGBRegressor

import warnings



# Ignore useless warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")



# Avoid runtime error messages

pd.set_option('display.float_format', lambda x:'%f'%x)



# make notebook's output stable across runs

np.random.seed(42)



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import os

print(os.listdir("../input"))
#!ls ../input/
#comando parse_dates: coloca data no formato yyyy-mm-dd

train_df = pd.read_csv("../input/elo-merchant-category-recommendation/train.csv", parse_dates=["first_active_month"])

test_df = pd.read_csv("../input/elo-merchant-category-recommendation/test.csv", parse_dates=["first_active_month"])

print("Número de linhas e colunas no train_df : ",train_df.shape)

print("Número de linhas e colunas no teste_df : ",test_df.shape)
# Get column names

column_names = train_df.columns

print(column_names)
# Get column data types

train_df.dtypes
# Also check if the column is unique

for i in column_names:

  print('{} is unique: {}'.format(i, train_df[i].is_unique))
# Check the index values

train_df.index.values
train_df.head()
train_df.isnull().sum().sum()
train_df.describe()
plt.figure(figsize=(12,8))

sns.distplot(train_df['target'].values, bins=50, kde=False, color="blue")

plt.title("Histogram of Loyalty score")

plt.xlabel('Loyalty score', fontsize=12)
(train_df['target'] < -30).sum()
#agrega todas as datas

cnt_srs = train_df['first_active_month'].dt.date.value_counts()

#faz um sort

cnt_srs = cnt_srs.sort_index() 

plt.figure(figsize=(14,6)) 

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='orange')

plt.xticks(rotation='vertical')

plt.xlabel('First active month', fontsize=12)

plt.ylabel('Número de cartões', fontsize=12)

plt.title("Contagem Número de cartões x first_active_month")
train_df.describe(include='O')
train_df["year"] = train_df["first_active_month"].dt.year

train_df["month"] = train_df["first_active_month"].dt.month

test_df["year"] = test_df["first_active_month"].dt.year

test_df["month"] = test_df["first_active_month"].dt.month
train_df1 = train_df
#all_df = pd.concat([train_df,test_df], axis=0, ignore_index=True)
hist_df = pd.read_csv("../input/elo-merchant-category-recommendation/historical_transactions.csv", nrows=5000000)

hist_df.head()
hist_df.shape
hist_df.dtypes
#verificar valores nulos

hist_df.isnull().sum()
hist_df.describe()
hist_df['card_id'].value_counts()
new_trans_df = pd.read_csv("../input/elo-merchant-category-recommendation/new_merchant_transactions.csv")

new_trans_df.head()
new_trans_df['card_id'].value_counts()
# Get column data types

new_trans_df.dtypes
new_trans_df.describe()
new_trans_df.isnull().sum()
new_trans_df.isnull().sum()
plt.figure(figsize=(12,8))

sns.distplot(new_trans_df['purchase_amount'].values, bins=50, kde=False, color="blue")

plt.title("Histogram of purchase_amount")

plt.xlabel('purchase_amount', fontsize=12)
new_trans_df.shape
new_trans_df.hist(bins=50, figsize=(20,15))

plt.tight_layout(pad=0.4)
train_df.shape
test_df.shape
newTransGr = pd.read_csv("../input/newtransgr2/newTransGr2.csv")
newTransGr.shape
newTransGr.columns
all_data1 = pd.merge(train_df, newTransGr, on='card_id', how='inner')
all_data2 = pd.merge(test_df, newTransGr, on='card_id', how='inner')
#all_datax = pd.merge(all_df, newTransGr, on='card_id', how='inner')
all_data1.shape
all_data2.shape
all_data2.columns
plt.figure(figsize=(12,8))

sns.distplot(newTransGr['purchase_amount'].values, bins=50, kde=False, color="blue")

plt.title("Histogram of listGroup")

plt.xlabel('listGroup', fontsize=12)
all_data1 = all_data1.drop(['card_id'], axis=1)

test_card = all_data2['card_id']

all_data2 = all_data2.drop(['card_id'], axis=1)
all_data1_dummies = pd.get_dummies(all_data1)
all_data1_dummies.head()
all_data2_dummies = pd.get_dummies(all_data2)
all_data1_dummies.columns
all_data2_dummies.columns
all_data1_dummies = all_data1_dummies.drop(['first_active_month'], axis=1)

#train_dummies = train_dummies.drop(['target'], axis=1)
all_data2_dummies = all_data2_dummies.drop(['first_active_month'], axis=1)
cols_to_use = ['feature_1', 'feature_2', 'feature_3', 'year',

       'month', 'City_Mode', 'Install_Mode', 'Merch_Mode', 'Mon_mean',

       'purchase_amount', 'Cat2_Mode', 'State_Mode', 'Subsec_Mode',

       'Cat1_Mode_N', 'Cat1_Mode_Y', 'Cat3_Mode_A', 'Cat3_Mode_B',

       'Cat3_Mode_C']



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



train_X = all_data1_dummies[cols_to_use]

test_X = all_data2_dummies[cols_to_use]

train_y = all_data1_dummies['target'].values



pred_test = 0

kf = model_selection.KFold(n_splits=5, random_state=2018, shuffle=True)

for dev_index, val_index in kf.split(all_data1_dummies):

    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    

    pred_test_tmp, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

    pred_test += pred_test_tmp

pred_test /= 5.
#fig, ax = plt.subplots(figsize=(12,10))

#lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

#ax.grid(False)

#plt.title("LightGBM - Feature Importance", fontsize=15)
pred_test.shape
test_card = pd.DataFrame({"card_id":test_card.values})

test_card
sub_df = pd.DataFrame({"card_id":test_card["card_id"].values})

sub_df["target"] = pred_test

print(sub_df)

sub_df.to_csv("baseline3_lgb.csv", index=False)
sub_df.shape