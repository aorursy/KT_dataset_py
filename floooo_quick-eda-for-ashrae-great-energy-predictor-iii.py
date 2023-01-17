from pathlib import Path



from IPython.display import display

import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from scipy.stats import norm, probplot
%matplotlib inline
INPUT_PATH = Path('/kaggle', 'input', 'ashrae-energy-prediction')

TRAIN_PATH = INPUT_PATH / 'train.csv'
df_train = pd.read_csv(TRAIN_PATH, parse_dates=['timestamp'])

df_train.head()
df_train['timestamp'].min(), df_train['timestamp'].max()
df_train.info()
display(df_train.describe().T)

display(pd.DataFrame([df_train[col].nunique() for col in df_train.columns],

             index=df_train.columns, columns=['nunique']))
df_train['timestamp'].min(), df_train['timestamp'].max()
ax = sns.distplot(df_train['building_id'].value_counts(),

                  kde=False, hist_kws=dict(density=True));

ax.set_title('Histogram of the frequencies of building_id');
METER_MAPPER = {0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'}
_, ax = plt.subplots()



df_train['meter'].value_counts().plot.bar(ax=ax)

ax.set_xlabel("Meter")

ax.set_ylabel('Count')

ax.set_xticklabels(METER_MAPPER.values());
ax = sns.countplot(df_train.groupby('building_id')['meter'].nunique())

ax.set_title("Frequencies of count of meter per building_id");
sns.distplot(df_train['meter_reading']);
df_train['meter_reading'].describe()
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

sns.distplot(np.log1p(df_train['meter_reading']), fit=norm, ax=ax1)

_ = probplot(np.log1p(df_train['meter_reading']), plot=ax2);
df_train.loc[df_train['meter_reading'] == 0].shape[0]
non_zero_meter_readings = df_train.loc[df_train['meter_reading'] != 0, 'meter_reading']



_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

sns.distplot(np.log1p(non_zero_meter_readings), fit=norm, ax=ax1)

_ = probplot(np.log1p(non_zero_meter_readings), plot=ax2);
df_train['log1p_meter_reading'] = np.log1p(df_train['meter_reading'])

df_train.head()
_, axes = plt.subplots(1, 2, figsize=(14, 4))



sns.barplot(x='meter', y='log1p_meter_reading', data=df_train, ax=axes[0])

sns.violinplot(x='meter', y='log1p_meter_reading', data=df_train, ax=axes[1])



for ax in axes:

    ax.set_xticklabels(METER_MAPPER.values());
frequencies_grouper = [

    ('H', 'hour'),

    ('D', 'day'),

    ('D', 'dayofweek'),

    ('W', 'week'),

    ('MS', 'month'),

]
for _, g in frequencies_grouper:

    df_train[g] = getattr(df_train['timestamp'].dt, g)

df_train.head()
fig = plt.figure(figsize=(18, 8), tight_layout=True)



gs = fig.add_gridspec(2, 6)



axes = np.array([

    fig.add_subplot(gs[0, :3]),

    fig.add_subplot(gs[0, 3:]),

    fig.add_subplot(gs[1, :2]),

    fig.add_subplot(gs[1, 2:4]),

    fig.add_subplot(gs[1, 4:]),

])



for (f, g), ax in zip(frequencies_grouper, axes):

    grouper = (df_train.groupby(pd.Grouper(freq=f, key='timestamp'))\

               .agg(average=('meter_reading', 'mean')))

    grouper[g] = getattr(grouper.index, g)

    sns.barplot(x=g, y='average', data=grouper, ax=ax)

    ax.set_xlabel(g)

    ax.set_ylabel(f"Average meter reading")