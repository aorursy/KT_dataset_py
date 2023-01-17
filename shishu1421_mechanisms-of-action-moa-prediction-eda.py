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



import plotly.express as px

from IPython.display import display

pd.options.display.max_columns = None

import random

import seaborn as sns

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train['dataset'] = 'train'

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

test['dataset'] = 'test'

df = pd.concat([train, test])
train.head()
test.head()
print('Shape of Training set :{} and Shape of Test set : {}'.format(train.shape,test.shape))
df.info()
sns.set(rc={'figure.figsize':(11.7,8.27)})

ds = df.groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_type', 'dataset', 'count']

sns.barplot(x='cp_type',

            hue='dataset',

            y='count',

            data=ds).set_title('Train/Test Count : Cp_type')
ds = df.groupby(['cp_time', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_time', 'dataset', 'count']

sns.barplot(x='cp_time',

            hue='dataset',

            y='count',

            data=ds).set_title('Train/Test Count : Cp_time')
ds = df.groupby(['cp_dose', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_dose', 'dataset', 'count']

sns.barplot(x='cp_dose',

            hue='dataset',

            y='count',

            data=ds).set_title('Train/Test Count : Cp_dose')
train_columns = train.columns.to_list()

g_list = [i for i in train_columns if i.startswith('g-')]

c_list = [i for i in train_columns if i.startswith('c-')]
import ipywidgets as widgets

from scipy import stats

from ipywidgets import interact, interact_manual

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.set(color_codes=True)

@interact

def distribution_colum(column = g_list):

    sns.distplot(train[column],kde=True)
import ipywidgets as widgets

from scipy import stats

from ipywidgets import interact, interact_manual

sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.set(color_codes=True)

@interact

def distribution_colum(column = c_list):

    sns.distplot(train[column],kde=True)
sns.set(rc={'figure.figsize':(19.7,15.27)})

columns = g_list + c_list

for_correlation = [columns[random.randint(0, len(columns)-1)] for i in range(40)]

data = train[for_correlation]

sns.heatmap(data.corr())
import itertools

cols = ['cp_time'] + columns

corr_matrix = train[cols].corr().abs()

high_corr_var=np.where(corr_matrix>0.9)

high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
sns.heatmap(train[list(set(itertools.chain(*high_corr_var)))].corr())
sns.set(rc={'figure.figsize':(10.7,8.27)})

@interact

def distribution_colum(column = list(set(itertools.chain(*high_corr_var)))):

    sns.distplot(train[column],kde=True)
train_target = pd.read_csv("../input/lish-moa/train_targets_scored.csv")



print('Number of rows : ', train_target.shape[0])

print('Number of cols : ', train_target.shape[1])

train_target.head()
x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values().reset_index()

x.columns = ['column', 'nonzero_records']

@interact 

def selct_number_of_columns(top_n = [10,50,100,150,200]):

    sns.barplot(x='nonzero_records',y='column',data=x.tail(top_n),orient='h')
sns.set(rc={'figure.figsize':(15.7,19.27)})

x = train_target.drop(['sig_id'], axis=1).sum(axis=0).sort_values(ascending=False).reset_index()

x.columns = ['column', 'count']

x['count'] = x['count'] * 100 / len(train_target)



sns.barplot(x='count',y='column',data=x.head(50)).set_title('Percent of positive records for every column in target')
data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = ['row', 'count']

data = data.groupby(['count'])['row'].count().reset_index()



fig = px.bar(

    data, 

    y=data['row'], 

    x="count", 

    title='Number of activations in targets for every sample', 

    width=800, 

    height=500

)

fig.show()
data = train_target.drop(['sig_id'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = ['row', 'count']

data = data.groupby(['count'])['row'].count().reset_index()

fig = px.pie(

    data, 

    values=100 * data['row']/len(train_target), 

    names="count", 

    title='Number of activations in targets for every sample (Percent)', 

    width=800, 

    height=500

)

fig.show()
train_target.describe()
import time

start = time.time()



correlation_matrix = pd.DataFrame()

for t_col in train_target.columns:

    corr_list = list()

    if t_col == 'sig_id':

        continue

    for col in columns:

        res = train[col].corr(train_target[t_col])

        corr_list.append(res)

    correlation_matrix[t_col] = corr_list

    

print(time.time()-start)
correlation_matrix['train_features'] = columns

correlation_matrix = correlation_matrix.set_index('train_features')

correlation_matrix
maxCol=lambda x: max(x.min(), x.max(), key=abs)

high_scores = correlation_matrix.apply(maxCol, axis=0).reset_index()

high_scores.columns = ['column', 'best_correlation']



fig = px.bar(

    high_scores, 

    x='column', 

    y="best_correlation", 

    orientation='v', 

    title='Best correlation with train columns for every target column', 

    width=1200,

    height=800

)

fig.show()
col_df = pd.DataFrame()

tr_cols = list()

tar_cols = list()

for col in correlation_matrix.columns:

    tar_cols.append(col)

    tr_cols.append(correlation_matrix[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(1).values[0])



col_df['column'] = tar_cols

col_df['train_best_column'] = tr_cols



total_scores = pd.merge(high_scores, col_df)

total_scores
count_features = total_scores['train_best_column'].value_counts().reset_index().sort_values('train_best_column')

count_features.columns = ['column', 'count']

fig = px.bar(

    count_features.tail(33), 

    x='count', 

    y="column", 

    orientation='h', 

    title='Columns from training set with number of high correlations with target columns', 

    width=800,

    height=700

)

fig.show()