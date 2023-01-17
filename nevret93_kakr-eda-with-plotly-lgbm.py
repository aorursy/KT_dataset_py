# LOAD LIBRARY

import gc

import os

import random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

plt.show()



import seaborn as sns

sns.set(style="whitegrid")



import warnings

warnings.filterwarnings('ignore')



# plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot



import cufflinks as cf

cf.go_offline()

cf.set_config_file(world_readable=True, theme='polar')



import lightgbm as lgb



from time import time

from tqdm import tqdm_notebook

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline
# SET RANDOM SEED

def seed_everything(seed=0):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)



SEED = 42

seed_everything(SEED)



# DATA CHECK

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# DEFINE BASE PATH

BASE_PATH = '/kaggle/input/kakr-4th-competition/'
# LOAD DATASET

df_train = pd.read_csv(os.path.join(BASE_PATH, 'train.csv'))

df_test  = pd.read_csv(os.path.join(BASE_PATH, 'test.csv'))
# CHECK TRAIN DATASET

print('Training data shape is: ', df_train.shape)

df_train.head(10)
# CHECK NULL VALUES AND DATA TYPES

print('Train dataset')

print(df_train.info())

print('')

print('')

print('Test dataset')

print(df_test.info())
print(df_train['id'].count())

print(df_train['id'].value_counts().shape[0])
df_train.groupby(['income']).count()['id'].to_frame()
df_train['income'].value_counts(normalize=True).iplot(kind='bar',

                                                      linecolor='black',

                                                      opacity=0.6,

                                                      color='red',

                                                      bargap=0.8,

                                                      gridcolor='white',

                                                      xTitle='Income',

                                                      yTitle='Percentage',

                                                      title='Distribution of the Target column in the training dataset')
df_train['age'].iplot(kind='hist',

                      bins=15,

                      color='yellow',

                      xTitle='age',

                      yTitle='Count',

                      title='Distribution of Final weight')
sns.kdeplot(df_train.loc[df_train['income'] == '<=50K', 'age'], label='<=50K', shade=True)

sns.kdeplot(df_train.loc[df_train['income'] == '>50K', 'age'], label='>50K', shade=True)



plt.xlabel('Age')

plt.ylabel('Density')
df_train['sex'].value_counts()
df_train['sex'].value_counts(normalize=True).iplot(kind='bar',

                                                   linecolor='black',

                                                   opacity=0.6,             

                                                   color='blue',

                                                   bargap=0.8,

                                                   gridcolor='white',

                                                   xTitle='Gender',

                                                   yTitle='Percentage',

                                                   title='Distribution of the Gender column in the training set')
df_gender_target = df_train.groupby(['income', 'sex'])['id'].count().to_frame().reset_index()

df_gender_target.style.background_gradient(cmap='Reds')
sns.catplot(x='income', y='id', hue='sex', data=df_gender_target, kind='bar')

plt.ylabel('Count')

plt.xlabel('Income')
df_train['race'].value_counts(normalize=True).sort_values(ascending=False)
df_train['race'].value_counts(normalize=True).iplot(kind='barh',

                                                    linecolor='black',

                                                    opacity=0.7,

                                                    color='orange',

                                                    bargap=0.8,

                                                    gridcolor='white',

                                                    xTitle='Percentage',

                                                    yTitle='Race',

                                                    title='Distribution of the race column in the training set')
df_race_target = df_train.groupby(['income', 'race'])['id'].count().to_frame().reset_index()

df_race_target.style.background_gradient(cmap='Reds')
sns.catplot(x='race', y='id', hue='income', data=df_race_target, kind='bar')



# plt.gcf().set_size_inches(10, 8)

plt.xlabel('Type of race')

plt.xticks(rotation=45, fontsize='10', horizontalalignment='right')

plt.ylabel('Count')
df_train['fnlwgt'].iplot(kind='hist',

                         bins=15,

                         color='green',

                         xTitle='fnlwgt',

                         yTitle='Count',

                         title='Distribution of Final weight')
sns.kdeplot(df_train.loc[df_train['income'] == '<=50K', 'fnlwgt'], label='<=50K', shade=True)

sns.kdeplot(df_train.loc[df_train['income'] == '>50K', 'fnlwgt'], label='>50K', shade=True)



plt.xlabel('Final weight')

plt.ylabel('Density')
df_train['hours_per_week'].iplot(kind='hist',

                                 bins=15,

                                 color='red',

                                 xTitle='Hours per week',

                                 yTitle='Count',

                                 title='Distribution of Hours per week')
sns.kdeplot(df_train.loc[df_train['income'] == '<=50K', 'hours_per_week'], label='<=50K', shade=True)

sns.kdeplot(df_train.loc[df_train['income'] == '>50K', 'hours_per_week'], label='>50K', shade=True)



plt.xlabel('Hours per week')

plt.ylabel('Density')
df_train['workclass'].value_counts()
df_train['workclass'].value_counts(normalize=True).sort_values().iplot(kind='barh',

                                                                       linecolor='black',

                                                                       opacity=0.7,

                                                                       color='skyblue',

                                                                       theme='pearl',

                                                                       bargap=0.2,

                                                                       gridcolor='white',

                                                                       xTitle='Percentage',

                                                                       yTitle='Workclass',

                                                                       title='Distribution of Workclass')
df_workclass_target = df_train.groupby(['workclass', 'income'])['id'].count().to_frame().reset_index()

df_workclass_target.style.background_gradient(cmap='Reds')
sns.catplot(x='workclass', y='id', hue='income', data=df_workclass_target, kind='bar')



# plt.gcf().set_size_inches(10, 8)

plt.xlabel('Type of workclass')

plt.xticks(rotation=45, fontsize='10', horizontalalignment='right')

plt.ylabel('Count')
df_train['income'] = np.where(df_train['income'] == '>50K', 1, 0)

y = df_train['income']

df_train = df_train.drop('income', axis=1)
# LABEL ENCODING

for col in df_train.columns:

    if df_train[col].dtype.name == 'object' or df_test[col].dtype.name == 'object':

        le = LabelEncoder()

        le.fit(list(df_train[col].values) + list(df_test[col].values))

        df_train[col] = le.transform(list(df_train[col].values))

        df_test[col]  = le.transform(list(df_test[col].values))
# reduce_mem_usage()

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2   

    

    for col in df.columns:

        col_type = df[col].dtypes

        

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

                    

    end_mem = df.memory_usage().sum() / 1024**2

    

    if verbose: 

        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

        

    return df
# REDUCE MEMORY USAGE

df_train = reduce_mem_usage(df_train)

df_test  = reduce_mem_usage(df_test)
X_train = df_train.drop(['id'], axis=1)

X_test  = df_test.drop(['id'], axis=1) 

y_train = y



print("X_train:", X_train.shape)

print("y_train:", y_train.shape)

print("X_test:",  X_test.shape)
def lgb_f1_score(y_hat, data):

    y_true = data.get_label()

    y_hat  = np.round(y_hat)

    

    return 'f1', f1_score(y_true, y_hat, average='weighted'), True
params = {

          'objective': 'binary',

          'max_depth': -1,

          'n_jobs': -1,

          'learning_rate': 0.01,

          'num_leaves': 50,

          'min_data_in_leaf': 30,

          'boosting_type': 'gbdt',

          'subsample_freq': 1,

          'subsample': 0.7,

          'n_estimators': 10000,

          'verbose': -1,

          'random_state': SEED,

          }
%%time

NFOLDS = 5

folds = KFold(n_splits=NFOLDS)



columns = X_train.columns

splits = folds.split(X_train, y_train)

y_preds = np.zeros(X_test.shape[0])

y_oof = np.zeros(X_train.shape[0])

score = 0



feature_importances = pd.DataFrame()

feature_importances['feature'] = columns



for fold_n, (trn_idx, val_idx) in enumerate(splits):

    X_trn, X_val = X_train[columns].iloc[trn_idx], X_train[columns].iloc[val_idx]

    y_trn, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    

    dtrain = lgb.Dataset(X_trn, label=y_trn)

    dvalid = lgb.Dataset(X_val, label=y_val)

    

    clf = lgb.train(

        params,

        dtrain,

        valid_sets = [dtrain, dvalid],

        verbose_eval = 200,

        early_stopping_rounds = 100,

        feval = lgb_f1_score

    )

    

    feature_importances[f'fold_{fold_n+1}'] = clf.feature_importance()

    

    y_pred_val = clf.predict(X_val) 

#     y_pred_val = np.where(y_pred_val >= 0.5, 1, 0)

    y_pred_val = [int(v >= 0.5) for v in y_pred_val]

    

    y_oof[val_idx] = y_pred_val

    print(f"Fold {fold_n + 1} | F1 Score: {f1_score(y_val, y_pred_val, average='weighted')}")

    

    score += f1_score(y_val, y_pred_val, average='weighted') / NFOLDS

    y_preds += clf.predict(X_test) / NFOLDS

    

    del X_trn, X_val, y_trn, y_val

    gc.collect()

    

print(f"\nMean F1 score = {score}")

print(f"OOF F1 score = {f1_score(y, y_oof, average='weighted')}")
sample_submission = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))

y_preds = np.where(y_preds >= 0.5, 1, 0)

sample_submission['prediction'] = y_preds



sample_submission.head(10)
sample_submission.to_csv('baseline_submission.csv', index=False)
feature_importances['average'] = feature_importances[[f'fold_{fold_n+1}' for fold_n in range(folds.n_splits)]].mean(axis=1)

feature_importances.to_csv('feature_importances.csv')



plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False), x='average', y='feature')

plt.title(f'Feature Importances over {folds.n_splits} folds average')