import warnings

warnings.simplefilter('ignore')

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import gc

import platform

import multiprocessing

import seaborn as sns

import sys

from tqdm import tqdm

pd.set_option('display.max_columns', 83)

pd.set_option('display.max_rows', 83)

plt.style.use('seaborn')

import os

import matplotlib

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import cufflinks

import plotly

for package in [pd, np, sns, matplotlib, plotly]:

    print(package.__name__, 'version:', package.__version__)

print('python version:', platform.python_version())

init_notebook_mode()
dtypes = {

    'Census_OSSkuName':                                     'category',

    'SkuEdition':                                           'category',

    'Census_OSEdition':                                     'category',

    'Platform':                                             'category',

    'OsVer':                                                'category',

    'Census_OSInstallLanguageIdentifier':                   'float16',

    'Census_OSUILocaleIdentifier':                          'int16',

    'Firewall':                                             'float16',

    'HasTpm':                                               'int8',

    'OsBuildLab':                                           'category',

    'Census_OSBuildNumber':                                 'int16',

    'OsBuild':                                              'int16',

    'Processor':                                            'category',

    'Census_OSArchitecture':                                'category',

    'Census_OSBranch':                                      'category',

    'OsPlatformSubRelease':                                 'category',

    'Census_MDC2FormFactor':                                'category',

    'Census_PowerPlatformRoleName':                         'category',

    'Census_ChassisTypeName':                               'category',

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
def label_encoder(df, features, keep_original=True, suffix='_label_encoded'):

    """

    Features should be list

    """

    for feature in tqdm(features):

        df[feature + suffix] = pd.factorize(df[feature])[0]

        if not keep_original:

            del df[feature]

    return df
df['Processor'].value_counts(dropna=False)
df['Census_OSArchitecture'].value_counts(dropna=False)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12,10))

sns.countplot(x='Processor', hue='HasDetections', data=train, ax=axes[0], order=['x64', 'arm64', 'x86']);

sns.countplot(x='Census_OSArchitecture', hue='HasDetections', data=train, ax=axes[1], order=['amd64', 'arm64', 'x86']);
df = label_encoder(df, ['Processor', 'Census_OSArchitecture'])

train = label_encoder(train, ['Processor', 'Census_OSArchitecture'])

df[['Processor_label_encoded', 'Census_OSArchitecture_label_encoded']].corr()
train[['Processor_label_encoded', 'Census_OSArchitecture_label_encoded', 'HasDetections']].corr()
df['OsPlatformSubRelease'].value_counts(dropna=False).head()
df['Census_OSBranch'].value_counts(dropna=False).head()
df = label_encoder(df, ['OsPlatformSubRelease', 'Census_OSBranch'])

train = label_encoder(train, ['OsPlatformSubRelease', 'Census_OSBranch'])

df[['OsPlatformSubRelease_label_encoded', 'Census_OSBranch_label_encoded']].corr()
df[['OsPlatformSubRelease_label_encoded', 'Census_OSBranch_label_encoded']].corr().columns
plt.figure(figsize=(14,6))

sns.countplot(x='OsPlatformSubRelease', hue='HasDetections', data=train);
os_platform_rs4 = train.loc[train['OsPlatformSubRelease'] == 'rs4', 'HasDetections'].value_counts().sort_index().to_dict()

os_branch_rs4 = train.loc[train['Census_OSBranch'] == 'rs4_release', 'HasDetections'].value_counts().sort_index().to_dict()

os_platform_rs2 = train.loc[train['OsPlatformSubRelease'] == 'rs2', 'HasDetections'].value_counts().sort_index().to_dict()

os_branch_rs2= train.loc[train['Census_OSBranch'] == 'rs2_release', 'HasDetections'].value_counts().sort_index().to_dict()

x = ['OsPlatformSubRelease', 'Census_OSBranch']

trace1 = go.Bar(x=x, y=[os_platform_rs4[0] / sum(os_platform_rs4.values()), os_branch_rs4[0] / sum(os_branch_rs4.values())], name='0 (no detections)', text=[os_platform_rs4[0], os_branch_rs4[0]], textposition="inside")

trace2 = go.Bar(x=x, y=[os_platform_rs4[1] / sum(os_platform_rs4.values()), os_branch_rs4[1] / sum(os_branch_rs4.values())], name='1 (has detections)', text=[os_platform_rs4[1], os_branch_rs4[1]], textposition="inside")

trace3 = go.Bar(x=x, y=[os_platform_rs2[0] / sum(os_platform_rs2.values()), os_branch_rs2[0] / sum(os_branch_rs2.values())], name='0 (no detections)', text=[os_platform_rs2[0], os_branch_rs2[0]], textposition="inside")

trace4 = go.Bar(x=x, y=[os_platform_rs2[1] / sum(os_platform_rs2.values()), os_branch_rs2[1] / sum(os_branch_rs2.values())], name='1 (has detections)', text=[os_platform_rs2[1], os_branch_rs2[1]], textposition="inside")

fig = plotly.tools.make_subplots(rows=1, cols=2, subplot_titles=('rs4', 'rs2'))

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig['layout'].update(title='rs4 and rs2 comparison', font=dict(size=18), barmode='group')

iplot(fig)
os_platform_rs3 = train.loc[train['OsPlatformSubRelease'] == 'rs3', 'HasDetections'].value_counts().sort_index().to_dict()

os_branch_rs3 = train.loc[train['Census_OSBranch'] == 'rs3_release', 'HasDetections'].value_counts().sort_index().to_dict()

os_branch_rs3_svc = train.loc[train['Census_OSBranch'] == 'rs3_release_svc_escrow', 'HasDetections'].value_counts().sort_index().to_dict()

trace1 = go.Bar(x=['OsPlatformSubRelease'], y=[os_platform_rs3[0] / sum(os_platform_rs3.values())], name='0 (no detections)', text=os_platform_rs3[0], textposition="inside")

trace2 = go.Bar(x=['OsPlatformSubRelease'], y=[os_platform_rs3[1] / sum(os_platform_rs3.values())], name='1 (has detections)', text=os_platform_rs3[1], textposition="inside")

trace3 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3[0] / sum(os_branch_rs3.values())], name='0 (no detections)', text=os_branch_rs3[0], textposition="inside")

trace4 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3[1] / sum(os_branch_rs3.values())], name='1 (has detections)', text=os_branch_rs3[1], textposition="inside")

trace5 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3_svc[0] / sum(os_branch_rs3_svc.values())], name='0 (no detections)', text=os_branch_rs3_svc[0], textposition="inside")

trace6 = go.Bar(x=['Census_OSBranch'], y=[os_branch_rs3_svc[1] / sum(os_branch_rs3_svc.values())], name='1 (has detections)', text=os_branch_rs3_svc[1], textposition="inside")

fig = plotly.tools.make_subplots(rows=1, cols=3, subplot_titles=('rs3', 'rs3_release', 'rs3_release_svc_escrow'))

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 1)

fig.append_trace(trace3, 1, 2)

fig.append_trace(trace4, 1, 2)

fig.append_trace(trace5, 1, 3)

fig.append_trace(trace6, 1, 3)

fig['layout'].update(title='rs3 comparison', font=dict(size=18), barmode='group')

iplot(fig)
df['OsVer'].value_counts(dropna=False).head()
df['Platform'].value_counts(dropna=False)
for i in df['Platform'].value_counts(dropna=False).index:

    print('Value counts for', i)

    print(df[df['Platform'] == i]['OsVer'].value_counts().head())

    print()
df = label_encoder(df, ['Platform', 'OsVer'])
df[['Platform_label_encoded', 'OsVer_label_encoded']].corr()
train = label_encoder(train, ['Platform', 'OsVer'])

train[['Platform_label_encoded', 'OsVer_label_encoded', 'HasDetections']].corr()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,10))

fig.subplots_adjust(wspace=0.2, hspace=0.4)

platforms = train['Platform'].value_counts(dropna=False).index

values_to_show = 3

for i in range(len(platforms)):

    sub_df = train[train['Platform'] == platforms[i]]['OsVer'].value_counts(normalize=True, dropna=False).head(values_to_show)

    sub_df.plot(kind='bar', rot=0, ax=axes[i // 2,i % 2], fontsize=14).set_xlabel('OsVer', fontsize=14);

    for j in range(values_to_show):

        try:

            axes[i // 2,i % 2].plot(j, train.loc[(train['Platform'] == platforms[i]) & (train['OsVer'] == sub_df.index[j]), 'HasDetections'].value_counts(True)[1], marker='.', color="black", markersize=24)

        except:

            continue

    axes[i // 2,i % 2].legend(['Detection rate (%)'])

    axes[i // 2,i % 2].set_title(platforms[i], fontsize=18)
df['Census_OSEdition'].value_counts(dropna=False).head()
df['Census_OSSkuName'].value_counts(dropna=False).head()
check_values = 5

fig, axes = plt.subplots(nrows=check_values, ncols=2, figsize=(14,check_values * 4))

fig.subplots_adjust(wspace=0.2, hspace=0.4)

for i in range(check_values):

    os_edition_value = train['Census_OSEdition'].value_counts(dropna=False).index[i]

    os_skuname_value = train['Census_OSSkuName'].value_counts(dropna=False).index[i]

    train.loc[train['Census_OSEdition'] == os_edition_value, 'HasDetections'].value_counts(True).sort_index().plot(kind='bar', rot=0, ax=axes[i, 0]).set_xlabel(os_edition_value, fontsize=16);

    train.loc[train['Census_OSSkuName'] == os_skuname_value, 'HasDetections'].value_counts(True).sort_index().plot(kind='bar', rot=0, ax=axes[i, 1]).set_xlabel(os_skuname_value, fontsize=16);

    axes[i, 0].text(x=-0.175, y=0.4, s=train.loc[train['Census_OSEdition'] == os_edition_value, 'HasDetections'].value_counts()[0], fontsize=18, color='white', fontweight='bold');

    axes[i, 0].text(x=0.825, y=0.4, s=train.loc[train['Census_OSEdition'] == os_edition_value, 'HasDetections'].value_counts()[1], fontsize=18, color='white', fontweight='bold');

    axes[i, 1].text(x=-0.175, y=0.4, s=train.loc[train['Census_OSSkuName'] == os_skuname_value, 'HasDetections'].value_counts()[0], fontsize=18, color='white', fontweight='bold');

    axes[i, 1].text(x=0.825, y=0.4, s=train.loc[train['Census_OSSkuName'] == os_skuname_value, 'HasDetections'].value_counts()[1], fontsize=18, color='white', fontweight='bold');

axes[0, 0].set_title('Census_OSEdition', fontsize=18, fontweight='bold');

axes[0, 1].set_title('Census_OSSkuName', fontsize=18, fontweight='bold');
train = label_encoder(train, ['Census_OSEdition', 'Census_OSSkuName'], keep_original=True)
train[['HasDetections', 'Census_OSEdition_label_encoded', 'Census_OSSkuName_label_encoded']].corr()
df['Census_OSInstallLanguageIdentifier'].value_counts().head()
df['Census_OSUILocaleIdentifier'].value_counts().head()
train['Census_OSInstallLanguageIdentifier'].fillna(-1, inplace=True)

train[['Census_OSInstallLanguageIdentifier', 'Census_OSUILocaleIdentifier', 'HasDetections']].corr()
train['Census_OSBuildNumber'].value_counts().head()
train['OsBuild'].value_counts().head()
train[['Census_OSBuildNumber', 'OsBuild', 'HasDetections']].corr()
diff = df[df['Census_OSBuildNumber'] != df['OsBuild']].shape[0]

print(diff, 'differences ({:<.3f} %)'.format(diff / df.shape[0]))
df['Census_PowerPlatformRoleName'].value_counts(dropna=False).head()
df['Census_MDC2FormFactor'].value_counts(dropna=False).head()
df['Census_ChassisTypeName'].value_counts(dropna=False).head()
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14,10))

sns.countplot(x='Census_PowerPlatformRoleName', hue='HasDetections', data=train, ax=axes[0], order=train['Census_PowerPlatformRoleName'].value_counts(dropna=False).index.tolist());

sns.countplot(x='Census_MDC2FormFactor', hue='HasDetections', data=train, ax=axes[1], order=train['Census_MDC2FormFactor'].value_counts(dropna=False).index.tolist());
train = label_encoder(train, ['Census_PowerPlatformRoleName', 'Census_MDC2FormFactor', 'Census_ChassisTypeName'], keep_original=True)
train[['HasDetections', 'Census_PowerPlatformRoleName_label_encoded', 'Census_MDC2FormFactor_label_encoded', 'Census_ChassisTypeName_label_encoded']].corr()