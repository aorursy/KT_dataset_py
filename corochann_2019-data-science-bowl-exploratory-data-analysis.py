from IPython.display import HTML, IFrame



HTML('<iframe width="400" height="200" src="https://pbskids.org/apps/media/video/Seesaw_v6_subtitled_ccmix.mp4" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
import gc

import os

from pathlib import Path

import random

import sys



from tqdm import tqdm_notebook as tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb
# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time

datadir = Path('/kaggle/input/data-science-bowl-2019')



# Read in the data CSV files

train = pd.read_csv(datadir/'train.csv')

train_labels = pd.read_csv(datadir/'train_labels.csv')

test = pd.read_csv(datadir/'test.csv')

specs = pd.read_csv(datadir/'specs.csv')

ss = pd.read_csv(datadir/'sample_submission.csv')
train["timestamp"] = pd.to_datetime(train["timestamp"])

test["timestamp"] = pd.to_datetime(test["timestamp"])
print(f'train.shape        : {train.shape}')

print(f'train_labels.shape : {train_labels.shape}')

print(f'test.shape         : {test.shape}')

print(f'specs.shape        : {specs.shape}')

print(f'ss.shape           : {ss.shape}')      
train.head()
train_labels.head()
train_labels['accuracy_group'].value_counts().sort_index().plot(kind="bar", title='accuracy group counts')
ss.head()
test.installation_id.nunique()
#101999d8

# f47ef997



test_tmp = test[test['installation_id'] == '101999d8']
test_tmp
ax = sns.distplot(test['installation_id'].value_counts().values)

ax.set_title('Number of event ids for each installation id')
train_labels[train_labels.installation_id == '0006a69f']
tmp_train = train[train.installation_id == '0006a69f']

tmp_train
tmp_train[tmp_train['event_code'].isin([4100, 4110]) & (tmp_train['type'] == 'Assessment')]
g = sns.FacetGrid(train_labels, col="title")

g = g.map(plt.hist, "accuracy_group")



# sns.distplot(train_labels, x='accuracy_group', hue='title')
train_labels['title'].value_counts().plot(kind="bar")
print('{} users solved {} Assessments in train_labels'

      .format(train_labels['installation_id'].nunique(), len(train_labels)))
sns.distplot(train_labels['installation_id'].value_counts().values)
target_id = '0006a69f'



px.scatter(train[train['installation_id'] == target_id], x='timestamp', y='event_code')
train.groupby(['title', 'type']).size().reset_index().rename(columns={0: 'count'}).sort_values('type')
specs.head()
specs.loc[0, 'args']