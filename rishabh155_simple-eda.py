import numpy as np

import pandas as pd

import os

from IPython.display import HTML

import json

import altair as alt

import matplotlib.pyplot as plt

import seaborn as sns



#%matplotlib inline

#from tqdm import tqdm_notebook

#from sklearn.preprocessing import StandardScaler

#from sklearn.svm import NuSVR, SVR

#from sklearn.metrics import mean_absolute_error

#pd.options.display.precision = 15



#import lightgbm as lgb

#import xgboost as xgb

#import time

#import datetime

#from catboost import CatBoostRegressor

#from sklearn.preprocessing import LabelEncoder

#from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

#from sklearn import metrics

#from sklearn import linear_model

#import gc



#import warnings

#warnings.filterwarnings("ignore")

#

#

#import networkx as nx



#alt.renderers.enable('notebook')



file_folder = '../input/champs-scalar-coupling' if 'champs-scalar-coupling' in os.listdir('../input/') else '../input'

os.listdir(file_folder)

train = pd.read_csv(f'{file_folder}/train.csv')

test = pd.read_csv(f'{file_folder}/test.csv')

sub = pd.read_csv(f'{file_folder}/sample_submission.csv')

structures = pd.read_csv(f'{file_folder}/structures.csv')

scc = pd.read_csv(f'{file_folder}/scalar_coupling_contributions.csv')



train.head()
structures.head()
scc.head()
print(f"Taining Data Set \n{train.nunique()}\n\n")

print(f"Unique atoms:{structures['atom'].nunique()}  {structures['atom'].unique()}")

print(f"Unique types: {train['type'].nunique()}  {train['type'].unique()}")
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}','x': f'x_{atom_idx}','y': f'y_{atom_idx}','z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)



train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)



grid = sns.FacetGrid(train[['type', 'dist']], col='type', hue='type', col_wrap=4)

grid.map(sns.distplot, 'dist')
train = pd.merge(train, scc, how = 'left',

                  left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

print("")

print(f"{train['atom_0'].unique()}")

print(f"{test['atom_0'].unique()}")

print(f"{train['atom_1'].unique()}")

print(f"{test['atom_1'].unique()}")

print(f"Unique types: {test['type'].nunique()}  {test['type'].unique()}")
plt.scatter(train['fc'],train['scalar_coupling_constant'])

plt.xlabel('fc')

plt.ylabel('scalar_coupling_constant')

plt.title('Correlation between SCC and fc')

plt.show()
fig, ax = plt.subplots(figsize = (20, 10))

for i, t in enumerate(train['type'].unique()):

    plt.subplot(2, 4, i + 1);

    plt.scatter(train.loc[train['type'] == t, 'fc'], train.loc[train['type'] == t, 'scalar_coupling_constant'], label=t);

    plt.title(f'fc vs target \n for {t} type');
train_new=train[['id','molecule_name','atom_index_0','atom_index_1','type','scalar_coupling_constant','dist','fc']].copy()

train_new.head()