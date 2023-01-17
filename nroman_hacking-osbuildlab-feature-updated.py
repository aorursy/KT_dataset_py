import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import gc
import multiprocessing
%matplotlib inline
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 83)
pd.set_option('display.max_rows', 20)
for package in [np, pd, sns]:
    print(package.__name__, 'version:', package.__version__)
import os
print(os.listdir("../input"))
dtypes = {
    'OsBuildLab':                                           'category',
    'Processor':                                            'category',
    'OsPlatformSubRelease':                                 'category',
    'Census_OSArchitecture':                                'category',
    'Census_OSBranch':                                      'category',
    'Census_OSBuildNumber':                                 'int16',
    'Census_OSBuildRevision':                               'int32',
    'OsBuild':                                              'int16',
    'HasDetections':                                        'int8'
}
def load_dataframe(dataset):
    usecols = dtypes.keys()
    if dataset == 'test':
        usecols = [col for col in dtypes.keys() if col != 'HasDetections']
    df = pd.read_csv(f'../input/{dataset}.csv', dtype=dtypes, usecols=usecols)
    return df
%%time
with multiprocessing.Pool() as pool: 
    train, test = pool.map(load_dataframe, ["train", "test"])
df = pd.concat([train.drop('HasDetections', axis=1), test])
df['OsBuildLab'].value_counts().head()
df['OsBuildReleaseYear'] = df['OsBuildLab'].str.slice(start=-11, stop=-9)
train['OsBuildReleaseYear'] = train['OsBuildLab'].str.slice(start=-11, stop=-9)
test['OsBuildReleaseYear'] = test['OsBuildLab'].str.slice(start=-11, stop=-9)
df['OsBuildReleaseYear'].value_counts(dropna=False).plot(kind='bar', figsize=(12,6), rot=0);
train['OsBuildReleaseYear'].value_counts(dropna=False).plot(kind='bar', figsize=(12,6), rot=0);
test['OsBuildReleaseYear'].value_counts(dropna=False).plot(kind='bar', figsize=(12,6), rot=0);
plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYear', hue='HasDetections', data=train);
df['OsBuildReleaseYearMonth'] = df['OsBuildLab'].str.slice(start=-11, stop=-7).astype('float16')
train['OsBuildReleaseYearMonth'] = train['OsBuildLab'].str.slice(start=-11, stop=-7).astype('float16')
test['OsBuildReleaseYearMonth'] = test['OsBuildLab'].str.slice(start=-11, stop=-7).astype('float16')
df['OsBuildReleaseYearMonth'].value_counts(dropna=False).head(10).plot(kind='bar', rot=0, figsize=(12,6));
train['OsBuildReleaseYearMonth'].value_counts(dropna=False).head(10).plot(kind='bar', rot=0, figsize=(12,6));
test['OsBuildReleaseYearMonth'].value_counts(dropna=False).head(10).plot(kind='bar', rot=0, figsize=(12,6));
plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYearMonth', hue='HasDetections', data=train[train['OsBuildReleaseYearMonth'] >= 1800]);
plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYearMonth', hue='HasDetections', data=train[(train['OsBuildReleaseYearMonth'] < 1800) & (train['OsBuildReleaseYearMonth'] >= 1700)]);
plt.subplots(figsize=(14,6))
sns.countplot(x='OsBuildReleaseYearMonth', hue='HasDetections', data=train[(train['OsBuildReleaseYearMonth'] < 1700) & (train['OsBuildReleaseYearMonth'] >= 1600)]);
del df
del test
del train['OsBuildReleaseYear'], train['OsBuildReleaseYearMonth']
gc.collect();
train = pd.concat([train, train['OsBuildLab'].str.replace('*', '.').str.split('.', expand=True)], axis=1)
train[0] = train[0].fillna(-1).astype('int16')
train[1] = train[1].fillna(-1).astype('int16')
train.head()
sns.heatmap(train.corr(), cmap="YlGnBu");
print(train[(train['Census_OSBuildNumber'] != train['OsBuild']) | (train['Census_OSBuildNumber'] != train[0]) | (train['OsBuild'] != train[0])][[0, 'OsBuild', 'Census_OSBuildNumber']].shape[0], 'differences')
train[(train['Census_OSBuildNumber'] != train['OsBuild']) | (train['Census_OSBuildNumber'] != train[0]) | (train['OsBuild'] != train[0])][[0, 'OsBuild', 'Census_OSBuildNumber']].head()
train[['OsBuild', 'Census_OSBuildNumber', 0]].corr()
train['Processor'].value_counts(dropna=False)
train['Census_OSArchitecture'].value_counts(dropna=False)
train[2].value_counts(dropna=False)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,12))
sns.countplot(x='Processor', hue='HasDetections', data=train, ax=axes[0], order=['x64', 'arm64', 'x86']);
sns.countplot(x='Census_OSArchitecture', hue='HasDetections', data=train, ax=axes[1], order=['amd64', 'arm64', 'x86']);
sns.countplot(x=2, hue='HasDetections', data=train, ax=axes[2], order=['amd64fre', 'arm64fre', 'x86fre']);
train[['OsPlatformSubRelease', 'Census_OSBranch', 3]].head(10)
train['OsPlatformSubRelease'].value_counts(dropna=False)
train['Census_OSBranch'].value_counts(dropna=False).head(10)
train[3].value_counts(dropna=False).head(10)
train.loc[train['OsPlatformSubRelease'] == 'rs3', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['green', 'orange']).set_xlabel('rs3', fontsize=18);
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
train.loc[train['Census_OSBranch'] == 'rs3_release', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['green', 'orange'], ax=axes[0]).set_xlabel('rs3_release', fontsize=18);
train.loc[train['Census_OSBranch'] == 'rs3_release_svc_escrow', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['orange', 'green'], ax=axes[1]).set_xlabel('rs3_release_svc_escrow', fontsize=18);
plt.gca().invert_xaxis();
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
train.loc[train[3] == 'rs3_release', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['green', 'orange'], ax=axes[0]).set_xlabel('rs3_release', fontsize=18);
train.loc[train[3] == 'rs3_release_svc_escrow', 'HasDetections'].value_counts(True).plot(kind='bar', rot=0, color=['orange', 'green'], ax=axes[1]).set_xlabel('rs3_release_svc_escrow', fontsize=18);
plt.gca().invert_xaxis();