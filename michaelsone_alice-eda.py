# Import libraries and set desired options

import numpy as np

import pandas as pd

from scipy.sparse import hstack

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import seaborn as sns

import eli5

from tqdm import tqdm

%config InlineBackend.figure_format = 'svg'

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# reading from csv

train_df = pd.read_csv('data/train_sessions.csv',

                       index_col='session_id', parse_dates=['time1'])

test_df = pd.read_csv('data/test_sessions.csv',

                      index_col='session_id', parse_dates=['time1'])



# Sort the data by time

train_df = train_df.sort_values(by='time1')



# Look at the first rows of the training set

train_df.head(3)
sites = ['site%s' % i for i in range(1, 11)]

times = ['time%s' % i for i in range(1, 11)]
train_df['year']=train_df['time1'].apply(lambda arr:arr.year)

train_df['hour']=train_df['time1'].apply(lambda arr:arr.hour)

train_df['day_of_week']=train_df['time1'].apply(lambda t: t.weekday())

train_df['month']=train_df['time1'].apply(lambda t: t.month)

sessduration = (train_df[times].apply(pd.to_datetime).max(axis=1) - train_df[times].apply(pd.to_datetime).min(axis=1)).astype('timedelta64[ms]').astype('int')

train_df['sessduration']=np.log1p(sessduration)

#train_df['sessduration'] = sessduration

#train_df['sessduration']=StandardScaler().fit_transform(sessduration.values.reshape(-1, 1))
train_df[train_df["target"]==1].head(5)
train_df.fillna(0).info()
train_df.describe()
# How many times it was Alice

train_df['target'].value_counts(normalize=False)
train_df.groupby('target').count()
train_df.head(5)
train_df.groupby('target')['sessduration','month','day_of_week','hour','year'].describe().T
train_df.fillna(0).tail(3)
plt.figure(figsize=(10, 4))

sns.distplot(train_df['sessduration'])
plt.figure(figsize=(10, 4))

sns.distplot(train_df[train_df['target']==1]['sessduration'])
train_df.columns
sns.countplot(x='day_of_week',data=train_df[train_df['target']==0])
sns.countplot(x='day_of_week',data=train_df[train_df['target']==1])
sns.countplot(x='hour',data=train_df[train_df['target']==0])
sns.countplot(x='hour',data=train_df[train_df['target']==1])
sns.countplot(x='year',data=train_df[train_df['target']==0])
sns.countplot(x='year',data=train_df[train_df['target']==1])
sns.countplot(x='month',data=train_df[train_df['target']==0])
sns.countplot(x='month',data=train_df[train_df['target']==1])
corr_matrix = train_df[['hour','day_of_week', 'month', 'sessduration']].corr()

sns.heatmap(corr_matrix,cmap='coolwarm');
plt.figure(figsize=(8, 4))

sns.boxplot(x='target',y='sessduration',data=train_df)
plt.figure(figsize=(8, 4))

sns.boxplot(x='target',y='hour',data=train_df)
plt.figure(figsize=(8, 4))

sns.boxplot(x='target',y='day_of_week',data=train_df)