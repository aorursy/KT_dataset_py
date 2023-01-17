# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import datetime

import lightgbm as lgb

from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler

import os

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import xgboost as xgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn import metrics

import json

import ast

import time

from sklearn import linear_model

import eli5

from eli5.sklearn import PermutationImportance

import shap

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from sklearn.neighbors import NearestNeighbors

from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE

import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape
test.shape
train.columns
train.info()
train.head()
train.describe()
train.dtypes.value_counts()
def missing_check(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    #print("Missing check:",missing_data )

    return missing_data
%time

missing_check(train)
%time

missing_check(test)
print('---------------------------------------------------------------------------------')

print(train['target'].value_counts())

print('---------------------------------------------------------------------------------')

print(train['target'].value_counts()/train['target'].shape[0])

print('---------------------------------------------------------------------------------')

#sns.set(style="darkgrid")

ax = sns.countplot(x=train['target'], data=train)
print("There are {}% target values with 1".format(100 * train["target"].value_counts()[1]/train.shape[0]))
#Histogram 

def Histogram(dataframe):

    fig, ax = plt.subplots(figsize=(20, 12))

    dataframe[dataframe.dtypes[(dataframe.dtypes=="float64")|(dataframe.dtypes=="int64")].index.values].hist(figsize=[11,11],bins=50, xlabelsize=10, ylabelsize=10,  ax=ax)

    plt.show()
Histogram(train.ix[:, 2:38])
Histogram(train.ix[:, 37:73])
Histogram(train.ix[:, 74:110])
Histogram(train.ix[:, 111:147])
Histogram(train.ix[:, 148:184])
Histogram(train.ix[:, 189:199])


col = ['var_0', 'var_1','var_2','var_3', 'var_4', 'var_5', 'var_6', 'var_7','var_8', 'var_9', 'var_10',

       'var_11','var_12', 'var_13', 'var_14', 'var_15', 'var_16', 'var_17', 'var_18','var_19', 'var_20' ]

       

for i in col:

       sns.catplot(x='target', y=i, data=train)


col = ['var_21', 'var_22','var_23', 'var_24', 'var_25', 'var_26', 'var_27','var_28', 'var_29', 'var_30',

       'var_31','var_32', 'var_33', 'var_34', 'var_35', 'var_36', 'var_37', 'var_38','var_39', 'var_40' ]

       

for i in col:

       sns.catplot(x='target', y=i, data=train)


col = ['var_41','var_42','var_43', 'var_44', 'var_45', 'var_46', 'var_47','var_48', 'var_49', 'var_50',

       'var_51','var_52', 'var_53', 'var_54', 'var_55', 'var_56', 'var_57', 'var_58','var_59', 'var_60' ]

       

for i in col:

       sns.catplot(x='target', y=i, data=train)


col = ['var_41','var_42','var_43', 'var_44', 'var_45', 'var_46', 'var_47','var_48', 'var_49', 'var_50',

       'var_51','var_52', 'var_53', 'var_54', 'var_55', 'var_56', 'var_57', 'var_58','var_59', 'var_60' ]

       

for i in col:

       sns.catplot(x='target', y=i, data=train)


col = ['var_61','var_62','var_63', 'var_64', 'var_65', 'var_66', 'var_67','var_68', 'var_69', 'var_70',

       'var_71','var_72', 'var_73', 'var_74', 'var_75', 'var_76', 'var_77', 'var_78','var_79', 'var_80' ]

       

for i in col:

       sns.catplot(x='target', y=i, data=train)
corr_df=train.iloc[:,1:].corr()

sns.set(style="whitegrid")

mask = np.zeros_like(corr_df.iloc[:,1:], dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_df.iloc[:,1:], mask=mask, cmap=cmap, vmax=.2, center=0,

            square=True, linewidths=.5)
corr = train.corr() 

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True)
corr_df.loc[corr_df.target<=-0.5].index[1:]

corr_target=corr_df.loc[corr_df.target>0.05]['target'].iloc[1:]

corr_target.plot(kind='bar')