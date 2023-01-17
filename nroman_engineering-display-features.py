import warnings

warnings.simplefilter('ignore')

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import gc

import multiprocessing

import seaborn as sns

pd.set_option('display.max_columns', 83)

pd.set_option('display.max_rows', 83)

plt.style.use('seaborn')

import os

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import cufflinks

import plotly

import matplotlib

init_notebook_mode()

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl', offline=True)

print(os.listdir("../input"))

for package in [pd, np, sns, matplotlib, plotly]:

    print(package.__name__, 'version:', package.__version__)
dtypes = {

        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',

        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',

        'Census_InternalPrimaryDisplayResolutionVertical':      'float16',

        'Census_IsTouchEnabled':                                'int8',

        'Census_IsPenCapable':                                  'int8',

        'Wdft_IsGamer':                                         'float16',

        'Processor':                                            'category',

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
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18,8))

train['Census_InternalPrimaryDisplayResolutionHorizontal'].value_counts().head(10).plot(kind='barh', ax=axes[0], fontsize=14).set_xlabel('Horizontal Resolution', fontsize=18)

train['Census_InternalPrimaryDisplayResolutionVertical'].value_counts().head(10).plot(kind='barh', ax=axes[1], fontsize=14).set_xlabel('Vertical Resolution', fontsize=18)

axes[0].invert_yaxis()

axes[1].invert_yaxis()
train['ResolutionRatio'] = train['Census_InternalPrimaryDisplayResolutionVertical'] / train['Census_InternalPrimaryDisplayResolutionHorizontal']
train['ResolutionRatio'].value_counts().head(10).plot(kind='barh', figsize=(14,8), fontsize=14);

plt.gca().invert_yaxis()
ratios = train['ResolutionRatio'].value_counts().head(6).index

fig, axes = plt.subplots(nrows=int(len(ratios) / 2), ncols=2, figsize=(16,14))

fig.subplots_adjust(wspace=0.2, hspace=0.4)

for i in range(len(ratios)):

    sns.countplot(x='ResolutionRatio', hue='HasDetections', data=train[train['ResolutionRatio'] == ratios[i]], ax=axes[i // 2,i % 2]);
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,10))

train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] < 720, 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[0,0]).set_xlabel('SD');

train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 720) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 1080), 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[0,1]).set_xlabel('HD');

train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 1080) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 2160), 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[1,0]).set_xlabel('FullHD');

train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] >= 2160, 'HasDetections'].value_counts().sort_index().plot(kind='bar', rot=0, ax=axes[1,1]).set_xlabel('4k');
sd_values = train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] < 720, 'HasDetections'].value_counts().sort_index().values

hd_values = train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 720) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 1080), 'HasDetections'].value_counts().sort_index().values

fullhd_values = train.loc[(train['Census_InternalPrimaryDisplayResolutionVertical'] >= 1080) & (train['Census_InternalPrimaryDisplayResolutionVertical'] < 2160), 'HasDetections'].value_counts().sort_index().values

k_values = train.loc[train['Census_InternalPrimaryDisplayResolutionVertical'] >= 2160, 'HasDetections'].value_counts().sort_index().values

x = ['SD', 'HD', 'FullHD', '4k']

y_0 = [sd_values[0], hd_values[0], fullhd_values[0], k_values[0]]

y_1 = [sd_values[1], hd_values[1], fullhd_values[1], k_values[1]]

trace1 = go.Bar(x=x, y=y_0, name='0 (no detections)')

trace2 = go.Bar(x=x, y=y_1, name='1 (has detections)')

data = [trace1, trace2]

layout = go.Layout(barmode='group')

fig = go.Figure(data=data, layout=layout)

iplot(fig)
def movingaverage(interval, window_size):

    window = np.ones(int(window_size))/float(window_size)

    return np.convolve(interval, window, 'same')



plot_dict = dict()

for i in train['Census_InternalPrimaryDiagonalDisplaySizeInInches'].value_counts().sort_index().index:

    try:

        plot_dict[i] = train.loc[train['Census_InternalPrimaryDiagonalDisplaySizeInInches'] == i, 'HasDetections'].value_counts(normalize=True)[1]

    except:

        plot_dict[i] = 0.0

fig, ax1 = plt.subplots(figsize=(16,7))

ax1.set_xlabel('Display Size in inches')

ax1.set_ylabel('Count', color='tab:green')

ax1.hist(plot_dict.keys(), color='tab:green', bins=int(len(plot_dict) / 20))

ax1.tick_params(axis='y', labelcolor='tab:green')

ax2 = ax1.twinx()

ax2.set_ylabel('Detection Rate', color='blue')

ax2.plot(plot_dict.keys(), movingaverage(list(plot_dict.values()), int(len(plot_dict) / 20)),color='blue', linewidth=2.0)

ax2.tick_params(axis='y', labelcolor='blue')

plt.show()
fig, axes = plt.subplots(nrows=int(len(ratios) / 2), ncols=2, figsize=(18,16))

fig.subplots_adjust(wspace=0.2, hspace=0.4)

for i in range(len(ratios)):

    train.loc[train['ResolutionRatio'] == ratios[i], 'Wdft_IsGamer'].value_counts(True, dropna=False).plot(kind='bar', rot=0, ax=axes[i // 2,i % 2], fontsize=14).set_xlabel('Wdft_IsGamer', fontsize=18)

    axes[i // 2,i % 2].plot(0, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Wdft_IsGamer'] == 0.0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

    axes[i // 2,i % 2].plot(1, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Wdft_IsGamer'] == 1.0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

    axes[i // 2,i % 2].plot(2, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Wdft_IsGamer'].isnull()), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24) 

    axes[i // 2,i % 2].legend(['Detection rate (%)'])

    axes[i // 2,i % 2].set_title('Ratio: ' + str(ratios[i]), fontsize=18)

fig.suptitle('Resolution rate to Wdft_IsGamer interaction', fontsize=18);
fig, axes = plt.subplots(nrows=int(len(ratios) / 2), ncols=2, figsize=(18,16))

fig.subplots_adjust(wspace=0.2, hspace=0.4)

for i in range(len(ratios)):

    train.loc[train['ResolutionRatio'] == ratios[i], 'Census_IsTouchEnabled'].value_counts(True, dropna=False).plot(kind='bar', rot=0, ax=axes[i // 2,i % 2], fontsize=14).set_xlabel('Census_IsTouchEnabled', fontsize=18)

    axes[i // 2,i % 2].plot(0, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Census_IsTouchEnabled'] == 0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

    axes[i // 2,i % 2].plot(1, train.loc[(train['ResolutionRatio'] == ratios[i]) & (train['Census_IsTouchEnabled'] == 1), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

    axes[i // 2,i % 2].legend(['Detection rate (%)'])

    axes[i // 2,i % 2].set_title('Ratio: ' + str(ratios[i]), fontsize=18)

fig.suptitle('Resolution rate to Census_IsTouchEnabled interaction', fontsize=18);
train['SD'] = (train['Census_InternalPrimaryDisplayResolutionVertical'] < 720).astype('uint8')

train['HD'] = (train['Census_InternalPrimaryDisplayResolutionVertical'].isin(range(720,1080))).astype('int8')

train['FullHD'] = (train['Census_InternalPrimaryDisplayResolutionVertical'].isin(range(1080,2160))).astype('int8')

train['4k'] = (train['Census_InternalPrimaryDisplayResolutionVertical'] >= 2160).astype('uint8')



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,12))

fig.subplots_adjust(wspace=0.2, hspace=0.4)

quals = ['SD', 'HD', 'FullHD', '4k']

axis_to_processor =  ['x86', 'x64', 'arm64']

for i in range(len(quals)):

    train.loc[train[quals[i]] == 1, 'Processor'].value_counts(True).sort_index(ascending=False).plot(kind='bar', rot=0, fontsize=14, ax=axes[i // 2, i % 2]).set_xlabel('Processor', fontsize=18);

    for j in range(len(axis_to_processor)):

        try:

            axes[i // 2,i % 2].plot(j, train.loc[(train[quals[i]] == 1) & (train['Processor'] == axis_to_processor[j]), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

        except:

            pass

    axes[i // 2,i % 2].legend(['Detection rate (%)'])

    axes[i // 2,i % 2].set_title('Display quality: ' + quals[i], fontsize=18)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18,12))

fig.subplots_adjust(wspace=0.2, hspace=0.4)

for i in range(len(quals)):

    train.loc[train[quals[i]] == 1, 'Wdft_IsGamer'].value_counts(True, dropna=False).sort_index().plot(kind='bar', rot=0, fontsize=14, ax=axes[i // 2, i % 2]).set_xlabel('Wdft_IsGamer', fontsize=18);

    axes[i // 2,i % 2].plot(0, train.loc[(train[quals[i]] == 1) & (train['Wdft_IsGamer'] == 0), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

    axes[i // 2,i % 2].plot(1, train.loc[(train[quals[i]] == 1) & (train['Wdft_IsGamer'] == 1), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

    axes[i // 2,i % 2].plot(2, train.loc[(train[quals[i]] == 1) & (train['Wdft_IsGamer'].isnull()), 'HasDetections'].value_counts(True, dropna=False)[1], marker='.', color="r", markersize=24)

    axes[i // 2,i % 2].legend(['Detection rate (%)'])

    axes[i // 2,i % 2].set_title('Display quality: ' + quals[i], fontsize=18)