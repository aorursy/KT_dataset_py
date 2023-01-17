# import packages

import os, gc

import numpy as np



# data manipulation

import pandas as pd

import pandas_profiling 



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import manifold

import cufflinks as cf

import plotly.offline



# Settings

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

plt.style.use('fivethirtyeight')

plt.show()



%matplotlib inline



os.listdir('../input/lish-moa/')
# root directory

ROOT = '../input/lish-moa/'



# files

target_scored = pd.read_csv(f'{ROOT}train_targets_scored.csv')

target_nonscored = pd.read_csv(f'{ROOT}train_targets_nonscored.csv')

train_features = pd.read_csv(f'{ROOT}train_features.csv')

test_features = pd.read_csv(f'{ROOT}test_features.csv')
train_features.head()
print(f'We have {train_features.shape[0]} rows and {train_features.shape[1]} columns in train_features.')
target_scored.head()
print(f'We have {target_scored.shape[0]} rows and {target_scored.shape[1]} columns in target_scored.')
target_nonscored.head()
print(f'We have {target_nonscored.shape[0]} rows and {target_nonscored.shape[1]} columns in target_nonscored.')
test_features.head()
print(f'We have {test_features.shape[0]} rows and {test_features.shape[1]} columns in test_features.')
# missing values

print(f'We have {train_features.isnull().values.sum()} missing values in train data')

print(f'We have {test_features.isnull().values.sum()} missing values in test data')
f = plt.figure(figsize=(16, 6))

gs = f.add_gridspec(1, 3)



with sns.axes_style("whitegrid"):

    ax = f.add_subplot(gs[0, 0])

    sns.countplot(train_features['cp_type'], palette="Set3")

    plt.title("Treatment Type")



with sns.axes_style("white"):

    ax = f.add_subplot(gs[0, 1])

    sns.countplot(train_features['cp_dose'], palette="Set2")

    plt.title("Treatment Dose")



with sns.axes_style("ticks"):

    ax = f.add_subplot(gs[0, 2])

    sns.countplot(train_features['cp_time'], palette="ch:.25")

    plt.title("Treatment Time")



f.tight_layout()
f = plt.figure(figsize=(16, 8))



with sns.axes_style("white"):

    for i in range(0,26):

        sns.kdeplot(train_features.loc[:,f"g-{i}"], shade=True);

        plt.title("Gene Distribution")
f = plt.figure(figsize=(20, 16))



mask = np.triu(np.ones_like(train_features.loc[:,"g-0":"g-771"].corr(), dtype=bool))

cmap = sns.diverging_palette(230, 20, as_cmap=True)



with sns.axes_style("white"):

    sns.heatmap(train_features.loc[:,"g-0":"g-771"].corr(), mask=mask, square=True, cmap=cmap);

    plt.title("Gene Expression: Correlation")
f = plt.figure(figsize=(16, 8))



with sns.axes_style("white"):

    for i in range(0,26):

        sns.kdeplot(train_features.loc[:,f"c-{i}"], shade=True);

        plt.title("Cell Viability Distribution")
f = plt.figure(figsize=(16, 10))



mask = np.triu(np.ones_like(train_features.loc[:,"c-0":"c-99"].corr(), dtype=bool))



with sns.axes_style("white"):

    sns.heatmap(train_features.loc[:,"c-0":"c-99"].corr(), mask=mask, square=True);

    plt.title("Cell Viability: Correlation")
data = target_scored.drop(['sig_id'], axis=1).sum().sort_values(inplace=False, ascending=False)

data = pd.Series(data)

with sns.axes_style("whitegrid"):

    data[:20].plot(kind='bar', figsize=(16, 8), title='Top 20 Target Features')
data = target_scored.drop(['sig_id'], axis=1).sum().sort_values(inplace=False)

data = pd.Series(data)

with sns.axes_style("whitegrid"):

    data[:20].plot(kind='bar', figsize=(16, 8), title='Last 20 Target Features')
with sns.axes_style("whitegrid"):

    target_scored.sum(axis=1).value_counts().plot(kind='bar', figsize=(16, 8), title='Number of MoA Activations per samples')
data = target_nonscored.drop(['sig_id'], axis=1).sum().sort_values(inplace=False, ascending=False)

data = pd.Series(data)

with sns.axes_style("whitegrid"):

    data[:20].plot(kind='bar', figsize=(16, 8), title='Top 20 Non Target Features')
with sns.axes_style("whitegrid"):

    target_nonscored.sum(axis=1).value_counts().plot(kind='bar', figsize=(16, 8), title='Number of MoA Activations per samples (Non Target Features)')