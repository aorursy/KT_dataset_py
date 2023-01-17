#Load libraries

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt



sns.set(font_scale=2.2)

plt.style.use('seaborn')



from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit

from sklearn.metrics import f1_score

import itertools

import lightgbm as lgb

import xgboost as xgb

from xgboost import XGBClassifier

import shap

from tqdm import tqdm

import featuretools as ft

import time

from datetime import date

import random 

import warnings

import operator



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.plotly as py

from plotly import tools

import plotly.figure_factory as ff



import warnings 

warnings.filterwarnings('ignore')

init_notebook_mode(connected=True)



#Load dataset

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print('Train Dataset shape:', df_train.shape)

print('Test Dataset shape shape: ', df_test.shape)
print ("Train Dataset: ")

df_train.head()
print ("Test Dataset: ")

df_test.head()
print ("Summary Statistics of Train Dataset: ")

df_train.describe()
print("Total Training Features with NaN values = " + str(df_train.columns[df_train.isnull().sum() != 0].size))

if (df_train.columns[df_train.isnull().sum() != 0].size):

    print("Features with NaN => {}".format(list(df_train.columns[df_train.isnull().sum() != 0])))

    df_train[df_train.columns[df_train.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
print ("Top Columns having missing values")

count = df_train.isnull().sum().sort_values(ascending = False)

percent = 100 * (df_train.isnull().sum() / df_train.isnull().count()).sort_values(ascending=False)

missing_df = pd.concat([count, percent], axis=1, keys=['Count', 'Percent'])

missing_df.head(5)

import missingno as msno

msno.matrix(df_train[['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']], color=(0.42, 0.6, 0.4))

plt.show()
# Value counts of target

print ("Value counts of target")

df_train_target_counts = df_train['Target'].value_counts().sort_index()

df_train_target_counts
# Value counts of target - bar plot

levels = ["Extereme Poverty", "Moderate Poverty", "Vulnerable", "Non vulnerable"]

trace = go.Bar(y=df_train_target_counts, x=levels, marker=dict(color='orange', opacity=0.6))

layout = dict(title="Household Poverty Levels", margin=dict(l=200), width=800, height=400)

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_train['idhogar'].nunique()
df_train['idhogar'].value_counts()
df_train['parentesco1'].value_counts()
# the subset with parentesco1 == 1

print ("Value counts of target")

df_train_head = df_train.loc[(df_train['Target'].notnull()) & (df_train['parentesco1'] == 1), ['Target', 'idhogar']]



# Value counts of target when parentesco1 == 1

df_train_target_counts = df_train_head['Target'].value_counts().sort_index()

df_train_target_counts
levels = ["Extereme Poverty", "Moderate Poverty", "Vulnerable", "Non vulnerable"]

trace = go.Bar(y=df_train_target_counts, x=levels, marker=dict(color='orange', opacity=0.6))

layout = dict(title="Household Poverty Levels", margin=dict(l=100), width=800, height=400)

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
df_train.info()
df_test.info()
# Count of Unique Values in Integer Columns

df_train_int_count = df_train.select_dtypes(np.int64).nunique().value_counts().sort_index()

df_train_int_count



trace = go.Bar(y=df_train_int_count, marker=dict(color='blue', opacity=0.8))

layout = dict(title="Count of Unique Values in Integer Columns", margin=dict(l=100), width=800, height=400,

              xaxis=dict(title='Number of Unique Values'), yaxis=dict(title="Count")

             )

data = [trace]

fig = go.Figure(data=data, layout=layout)

iplot(fig)
# distributions of the float columns by the target 

from collections import OrderedDict # fix the keys and values in the same order



plt.figure(figsize = (20, 16))

plt.style.use('fivethirtyeight')



# Color mapping

colors = OrderedDict({1: 'red', 2: 'orange', 3: 'purple', 4: 'green'})

poverty_mapping = OrderedDict({1: 'Extreme', 2: 'Moderate', 3: 'Vulnerable', 4: 'Non Vulnerable'})



# Iterate through the float columns

for i, col in enumerate(df_train.select_dtypes('float')):

    ax = plt.subplot(4, 2, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(df_train.loc[df_train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)
df_train.select_dtypes('object').head()
mapping = {"yes": 1, "no": 0}



# Apply same operation to both train and test

for df in [df_train, df_test]:

    # Fill in the values with the correct mapping

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64)

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)



df_train[['dependency', 'edjefa', 'edjefe']].describe()
plt.figure(figsize = (16, 12))



# Iterate through the float columns

for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):

    ax = plt.subplot(3, 1, i + 1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(df_train.loc[df_train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'{col.capitalize()} Distribution'); plt.xlabel(f'{col}'); plt.ylabel('Density')



plt.subplots_adjust(top = 2)
# Add null Target column to test

df_test['Target'] = np.nan

data = df_train.append(df_test, ignore_index = True)