# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing Libraries



#data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import math



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.offline as py



# machine learning

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, roc_auc_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

import gc

from sklearn import metrics

import lightgbm as lgb

#import xgboost as xgb

import time

import datetime

#from numba import jit
data_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

data_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')
data_train.head()
data_train.info()
data_test.info()
data_train.describe()
# Good Practice: Always check if data set is balance or imbalance.

sns.set_style('whitegrid')

sns.countplot(x='TARGET',data=data_train,palette='RdBu_r')
statistics_of_data = []

for col in data_train.columns:

  statistics_of_data.append((col,

                             data_train[col].nunique(),

                             data_train[col].isnull().sum()*100/data_train.shape[0],

                             data_train[col].value_counts(normalize=True, dropna=False).values[0] * 100, 

                             data_train[col].dtype

                             ))

stats_df = pd.DataFrame(statistics_of_data, columns=['Feature', 'Uniq_val', 'missing_val', 'val_biggest_cat', 'type'])
stats_df.sort_values('missing_val', ascending=False)
def exploreFeatures(col):

  top_n=10

  top_n = top_n if data_train[col].nunique() > top_n else data_train[col].nunique()

  print(f"{col} has {data_train[col].nunique()} unique values and type: {data_train[col].dtype}.")

  print(data_train[col].value_counts(normalize=True, dropna=False).head())
exploreFeatures('REG_REGION_NOT_WORK_REGION')
exploreFeatures('FLAG_MOBIL')
exploreFeatures('FLAG_DOCUMENT_12')
exploreFeatures('FLAG_DOCUMENT_10')
exploreFeatures('FLAG_DOCUMENT_2')
exploreFeatures('FLAG_DOCUMENT_4')
exploreFeatures('FLAG_DOCUMENT_7')
exploreFeatures('FLAG_DOCUMENT_17')
exploreFeatures('FLAG_DOCUMENT_21')
exploreFeatures('FLAG_DOCUMENT_20')
exploreFeatures('FLAG_OWN_CAR')
#Convert it into 0 and 1

#mapping = {N = 0,Y=1}

combine = [data_train,data_test]

titlemapping = {'N':0, 'Y':1}

for row in combine:

    row['FLAG_OWN_CAR'] = row['FLAG_OWN_CAR'].map(titlemapping)

    row['FLAG_OWN_CAR'] = row['FLAG_OWN_CAR'].fillna(0)
def GraphPlotsForEDA(col, only_bars=False, top_n=10, Has_Car=False):

    top_n = top_n if data_train[col].nunique() > top_n else data_train[col].nunique()

    #print(f"{col} has {train[col].nunique()} unique values and type: {train[col].dtype}.")

    #print(train[col].value_counts(normalize=True, dropna=False).head())

    if not Has_Car:

        if not only_bars:

            df = data_train.groupby([col]).agg({'TARGET': ['count', 'mean']})

            df = df.sort_values(('TARGET', 'count'), ascending=False).head(top_n).sort_index()

            data = [go.Bar(x=df.index, y=df['TARGET']['count'].values, name='counts'),

                    go.Scatter(x=df.index, y=df['TARGET']['mean'], name='Rate', yaxis='y2')]



            layout = go.Layout(dict(title = f"Counts of {col} by top-{top_n} categories and mean target value",

                                xaxis = dict(title = f'{col}',

                                             showgrid=False,

                                             zeroline=False,

                                             showline=False,),

                                yaxis = dict(title = 'Counts',

                                             showgrid=False,

                                             zeroline=False,

                                             showline=False,),

                                yaxis2=dict(title='Detections rate', overlaying='y', side='right')),

                           legend=dict(orientation="v"))



        else:

            top_cat = list(data_train[col].value_counts(dropna=False).index[:top_n])

            df0 = data_train.loc[(data_train[col].isin(top_cat)) & (data_train['TARGET'] == 1), col].value_counts().head(10).sort_index()

            df1 = data_train.loc[(data_train[col].isin(top_cat)) & (data_train['TARGET'] == 0), col].value_counts().head(10).sort_index()

            data = [go.Bar(x=df0.index, y=df0.values, name='Has a Car'),

                    go.Bar(x=df1.index, y=df1.values, name='Doesnt has a Car')]



            layout = go.Layout(dict(title = f"Counts of {col} by top-{top_n} categories",

                                xaxis = dict(title = f'{col}',

                                             showgrid=False,

                                             zeroline=False,

                                             showline=False,),

                                yaxis = dict(title = 'Counts',

                                             showgrid=False,

                                             zeroline=False,

                                             showline=False,),

                                ),

                           legend=dict(orientation="v"), barmode='group')

        

        py.iplot(dict(data=data, layout=layout))

        

    else:

        top_n = 10

        top_cat = list(data_train[col].value_counts(dropna=False).index[:top_n])

        df = data_train.loc[data_train[col].isin(top_cat)]



        df1 = data_train.loc[data_train['FLAG_OWN_CAR'] == 1]

        df0 = data_train.loc[data_train['FLAG_OWN_CAR'] == 0]



        df0_ = df0.groupby([col]).agg({'TARGET': ['count', 'mean']})

        df0_ = df0_.sort_values(('TARGET', 'count'), ascending=False).head(top_n).sort_index()

        df1_ = df1.groupby([col]).agg({'TARGET': ['count', 'mean']})

        df1_ = df1_.sort_values(('TARGET', 'count'), ascending=False).head(top_n).sort_index()

        data1 = [go.Bar(x=df0_.index, y=df0_['TARGET']['count'].values, name='Doesnot have Car counts'),

                go.Scatter(x=df0_.index, y=df0_['TARGET']['mean'], name='Return Rate for doesnot have a Car', yaxis='y2')]

        data2 = [go.Bar(x=df1_.index, y=df1_['TARGET']['count'].values, name='Does hava a Car counts'),

                go.Scatter(x=df1_.index, y=df1_['TARGET']['mean'], name='Return Rate for have a Car', yaxis='y2')]



        layout = go.Layout(dict(title = f"Counts of {col} by top-{top_n} categories for Doesnot have a Car",

                            xaxis = dict(title = f'{col}',

                                         showgrid=False,

                                         zeroline=False,

                                         showline=False,

                                         type='category'),

                            yaxis = dict(title = 'Counts',

                                         showgrid=False,

                                         zeroline=False,

                                         showline=False,),

                                    yaxis2=dict(title='Rate', overlaying='y', side='right'),

                            ),

                       legend=dict(orientation="v"), barmode='group')



        py.iplot(dict(data=data1, layout=layout))

        layout['title'] = f"Counts of {col} by top-{top_n} categories for Have a car"

        py.iplot(dict(data=data2, layout=layout))
GraphPlotsForEDA('AMT_INCOME_TOTAL',True,Has_Car=True)
GraphPlotsForEDA('CODE_GENDER',True)
GraphPlotsForEDA('FLAG_OWN_REALTY',True)
GraphPlotsForEDA('CNT_CHILDREN',True,Has_Car=True)
GraphPlotsForEDA('AMT_CREDIT',True,Has_Car = True)
GraphPlotsForEDA('AMT_ANNUITY',True,Has_Car = True)
GraphPlotsForEDA('AMT_GOODS_PRICE',True,Has_Car = True)
GraphPlotsForEDA('NAME_INCOME_TYPE',True)
GraphPlotsForEDA('NAME_EDUCATION_TYPE',True)
GraphPlotsForEDA('NAME_FAMILY_STATUS',True)
#Deleting these features having unique value > 98%

for col in data_train.columns:

    rate = data_train[col].value_counts(normalize=True, dropna=False).values[0]

    if rate > 0.98:

        #data_train.drop(col)

        print(col)
combine = [data_train, data_test]

print("Before", data_train.shape, data_test.shape,combine[0].shape, combine[1].shape)



data_train = data_train.drop([

'FLAG_MOBIL',

'NAME_CONTRACT_TYPE',

'EMERGENCYSTATE_MODE',

'FLAG_CONT_MOBILE',

'REG_REGION_NOT_LIVE_REGION',

'FLAG_DOCUMENT_2',

'FLAG_DOCUMENT_4',

'FLAG_DOCUMENT_5',

'FLAG_DOCUMENT_7',

'FLAG_DOCUMENT_9',

'FLAG_DOCUMENT_10',

'FLAG_DOCUMENT_11',

'FLAG_DOCUMENT_12',

'FLAG_DOCUMENT_13',

'FLAG_DOCUMENT_14',

'FLAG_DOCUMENT_15',

'FLAG_DOCUMENT_16',

'FLAG_DOCUMENT_17',

'FLAG_DOCUMENT_18',

'FLAG_DOCUMENT_19',

'FLAG_DOCUMENT_20',

'FLAG_DOCUMENT_21'

], axis=1)

data_test = data_test.drop([

'FLAG_MOBIL',

'FLAG_CONT_MOBILE',

'NAME_CONTRACT_TYPE',

'EMERGENCYSTATE_MODE',

'REG_REGION_NOT_LIVE_REGION',

'FLAG_DOCUMENT_2',

'FLAG_DOCUMENT_4',

'FLAG_DOCUMENT_5',

'FLAG_DOCUMENT_7',

'FLAG_DOCUMENT_9',

'FLAG_DOCUMENT_10',

'FLAG_DOCUMENT_11',

'FLAG_DOCUMENT_12',

'FLAG_DOCUMENT_13',

'FLAG_DOCUMENT_14',

'FLAG_DOCUMENT_15',

'FLAG_DOCUMENT_16',

'FLAG_DOCUMENT_17',

'FLAG_DOCUMENT_18',

'FLAG_DOCUMENT_19',

'FLAG_DOCUMENT_20',

'FLAG_DOCUMENT_21'

], axis=1)

combine = [data_train, data_test]



print("After", data_train.shape, data_test.shape, combine[0].shape, combine[1].shape)
#Convert it into 0 and 1

#mapping = {N = 0,Y=1}

combine = [data_train,data_test]

titlemapping = {'F':0, 'M':1}

for row in combine:

    row['CODE_GENDER'] = row['CODE_GENDER'].map(titlemapping)

    row['CODE_GENDER'] = row['CODE_GENDER'].fillna(0)
#Convert it into 0 and 1

#mapping = {N = 0,Y=1}

combine = [data_train,data_test]

titlemapping = {'N':0, 'Y':1}

for row in combine:

    row['FLAG_OWN_REALTY'] = row['FLAG_OWN_REALTY'].map(titlemapping)

    row['FLAG_OWN_REALTY'] = row['FLAG_OWN_REALTY'].fillna(0)
binary_variables = [c for c in data_train.columns if data_train[c].nunique() == 2]
binary_variables
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



newdf = data_train.select_dtypes(include=numerics)
newdf
trueNumericCol_and_categorical = [c for c in newdf.columns 

                       if (c not in binary_variables)]
trueNumericCol_and_categorical
data_train['CNT_CHILDREN'] = data_train['CNT_CHILDREN'].astype('category')

data_test['CNT_CHILDREN'] = data_test['CNT_CHILDREN'].astype('category')



data_train['OWN_CAR_AGE'] = data_train['OWN_CAR_AGE'].astype('category')

data_test['OWN_CAR_AGE'] = data_test['OWN_CAR_AGE'].astype('category')



data_train['CNT_FAM_MEMBERS'] = data_train['CNT_FAM_MEMBERS'].astype('category')

data_test['CNT_FAM_MEMBERS'] = data_test['CNT_FAM_MEMBERS'].astype('category')



data_train['REGION_RATING_CLIENT'] = data_train['REGION_RATING_CLIENT'].astype('category')

data_test['REGION_RATING_CLIENT'] = data_test['REGION_RATING_CLIENT'].astype('category')



data_train['REGION_RATING_CLIENT_W_CITY'] = data_train['REGION_RATING_CLIENT_W_CITY'].astype('category')

data_test['REGION_RATING_CLIENT_W_CITY'] = data_test['REGION_RATING_CLIENT_W_CITY'].astype('category')







data_train['HOUR_APPR_PROCESS_START'] = data_train['HOUR_APPR_PROCESS_START'].astype('category')

data_test['HOUR_APPR_PROCESS_START'] = data_test['HOUR_APPR_PROCESS_START'].astype('category')



data_train['OBS_30_CNT_SOCIAL_CIRCLE'] = data_train['OBS_30_CNT_SOCIAL_CIRCLE'].astype('category')

data_test['OBS_30_CNT_SOCIAL_CIRCLE'] = data_test['OBS_30_CNT_SOCIAL_CIRCLE'].astype('category')



data_train['DEF_30_CNT_SOCIAL_CIRCLE'] = data_train['DEF_30_CNT_SOCIAL_CIRCLE'].astype('category')

data_test['DEF_30_CNT_SOCIAL_CIRCLE'] = data_test['DEF_30_CNT_SOCIAL_CIRCLE'].astype('category')



data_train['OBS_60_CNT_SOCIAL_CIRCLE'] = data_train['OBS_60_CNT_SOCIAL_CIRCLE'].astype('category')

data_test['OBS_60_CNT_SOCIAL_CIRCLE'] = data_test['OBS_60_CNT_SOCIAL_CIRCLE'].astype('category')



data_train['DEF_60_CNT_SOCIAL_CIRCLE'] = data_train['DEF_60_CNT_SOCIAL_CIRCLE'].astype('category')

data_test['DEF_60_CNT_SOCIAL_CIRCLE'] = data_test['DEF_60_CNT_SOCIAL_CIRCLE'].astype('category')



data_train['DAYS_LAST_PHONE_CHANGE'] = data_train['DAYS_LAST_PHONE_CHANGE'].astype('category')

data_test['DAYS_LAST_PHONE_CHANGE'] = data_test['DAYS_LAST_PHONE_CHANGE'].astype('category')



data_train['AMT_REQ_CREDIT_BUREAU_HOUR'] = data_train['AMT_REQ_CREDIT_BUREAU_HOUR'].astype('category')

data_test['AMT_REQ_CREDIT_BUREAU_HOUR'] = data_test['AMT_REQ_CREDIT_BUREAU_HOUR'].astype('category')



data_train['AMT_REQ_CREDIT_BUREAU_DAY'] = data_train['AMT_REQ_CREDIT_BUREAU_DAY'].astype('category')

data_test['AMT_REQ_CREDIT_BUREAU_DAY'] = data_test['AMT_REQ_CREDIT_BUREAU_DAY'].astype('category')



data_train['AMT_REQ_CREDIT_BUREAU_WEEK'] = data_train['AMT_REQ_CREDIT_BUREAU_WEEK'].astype('category')

data_test['AMT_REQ_CREDIT_BUREAU_WEEK'] = data_test['AMT_REQ_CREDIT_BUREAU_WEEK'].astype('category')



data_train['AMT_REQ_CREDIT_BUREAU_MON'] = data_train['AMT_REQ_CREDIT_BUREAU_MON'].astype('category')

data_test['AMT_REQ_CREDIT_BUREAU_MON'] = data_test['AMT_REQ_CREDIT_BUREAU_MON'].astype('category')



data_train['AMT_REQ_CREDIT_BUREAU_QRT'] = data_train['AMT_REQ_CREDIT_BUREAU_QRT'].astype('category')

data_test['AMT_REQ_CREDIT_BUREAU_QRT'] = data_test['AMT_REQ_CREDIT_BUREAU_QRT'].astype('category')



data_train['AMT_REQ_CREDIT_BUREAU_YEAR'] = data_train['AMT_REQ_CREDIT_BUREAU_YEAR'].astype('category')

data_test['AMT_REQ_CREDIT_BUREAU_YEAR'] = data_test['AMT_REQ_CREDIT_BUREAU_YEAR'].astype('category')

true_numerical_columns = [

 'SK_ID_CURR',

 'AMT_INCOME_TOTAL',

 'AMT_CREDIT',

 'AMT_ANNUITY',

 'AMT_GOODS_PRICE',

 'REGION_POPULATION_RELATIVE',

 'DAYS_BIRTH',

 'DAYS_EMPLOYED',

 'DAYS_REGISTRATION',

 'DAYS_ID_PUBLISH',

 'EXT_SOURCE_1',

 'EXT_SOURCE_2',

 'EXT_SOURCE_3',

 'APARTMENTS_AVG',

 'BASEMENTAREA_AVG',

 'YEARS_BEGINEXPLUATATION_AVG',

 'YEARS_BUILD_AVG',

 'COMMONAREA_AVG',

 'ELEVATORS_AVG',

 'ENTRANCES_AVG',

 'FLOORSMAX_AVG',

 'FLOORSMIN_AVG',

 'LANDAREA_AVG',

 'LIVINGAPARTMENTS_AVG',

 'LIVINGAREA_AVG',

 'NONLIVINGAPARTMENTS_AVG',

 'NONLIVINGAREA_AVG',

 'APARTMENTS_MODE',

 'BASEMENTAREA_MODE',

 'YEARS_BEGINEXPLUATATION_MODE',

 'YEARS_BUILD_MODE',

 'COMMONAREA_MODE',

 'ELEVATORS_MODE',

 'ENTRANCES_MODE',

 'FLOORSMAX_MODE',

 'FLOORSMIN_MODE',

 'LANDAREA_MODE',

 'LIVINGAPARTMENTS_MODE',

 'LIVINGAREA_MODE',

 'NONLIVINGAPARTMENTS_MODE',

 'NONLIVINGAREA_MODE',

 'APARTMENTS_MEDI',

 'BASEMENTAREA_MEDI',

 'YEARS_BEGINEXPLUATATION_MEDI',

 'YEARS_BUILD_MEDI',

 'COMMONAREA_MEDI',

 'ELEVATORS_MEDI',

 'ENTRANCES_MEDI',

 'FLOORSMAX_MEDI',

 'FLOORSMIN_MEDI',

 'LANDAREA_MEDI',

 'LIVINGAPARTMENTS_MEDI',

 'LIVINGAREA_MEDI',

 'NONLIVINGAPARTMENTS_MEDI',

 'NONLIVINGAREA_MEDI',

 'TOTALAREA_MODE']
binary_variables
categorical_columns = [c for c in data_train.columns 

                       if (c not in true_numerical_columns) & (c not in binary_variables)]
categorical_columns
print(data_train.shape)

print(data_test.shape)
exploreFeatures('NAME_INCOME_TYPE')
#Frequency Encoding



from tqdm import tqdm

from tqdm import tqdm_notebook

#from sklearn.preprocessing import LabelEncoder

def frequency_encoding(variable):

    t = pd.concat([data_train[variable], data_test[variable]]).value_counts().reset_index()

    t = t.reset_index()

    t.loc[t[variable] == 1, 'level_0'] = np.nan

    t.set_index('index', inplace=True)

    max_label = t['level_0'].max() + 1

    t.fillna(max_label, inplace=True)

    return t.to_dict()['level_0']
for variable in tqdm(categorical_columns):

  freq_enc_dict = frequency_encoding(variable)

  data_train[variable] = data_train[variable].map(lambda x: freq_enc_dict.get(x, np.nan))

  data_test[variable] = data_test[variable].map(lambda x: freq_enc_dict.get(x, np.nan))

  #categorical_columns.remove(variable)
print(data_train.shape)

print(data_test.shape)
categorical_columns = [c for c in data_train.columns 

                       if (c not in true_numerical_columns) & (c not in binary_variables)]
train = data_train.copy()

test = data_test.copy()
target = train['TARGET']

del train['TARGET']
train = train.drop(["SK_ID_CURR"],axis =1)
#basic parameter:

parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 0

}
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

features = [c for c in train.columns if c not in ['SK_ID_CURR']]

folds = KFold(n_splits=5, shuffle=True, random_state=15)

categorical_columns = [c for c in categorical_columns if c not in ['SK_ID_CURR']]

predictions = np.zeros(len(test))

oof = np.zeros(len(train))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):

    print("fold n°{}".format(fold_))

    trn_data = lgb.Dataset(train.iloc[trn_idx][features],label=target.iloc[trn_idx],categorical_feature = categorical_columns)

    val_data = lgb.Dataset(train.iloc[val_idx][features],

                           label=target.iloc[val_idx],

                           categorical_feature = categorical_columns

                          )

    
import lightgbm

model = lightgbm.train(parameters,

                       trn_data,

                       valid_sets = [trn_data, val_data],

                       num_boost_round=5000,

                       early_stopping_rounds=100)
ax = lightgbm.plot_importance(model, max_num_features=70, figsize=(15,15))

plt.show()