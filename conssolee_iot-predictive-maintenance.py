# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

rul_df = pd.read_csv('../input/RUL.csv')



sensor_columns = [col for col in train_df.columns if col.startswith("sensor")]

setting_columns = [col for col in train_df.columns if col.startswith("setting")]
train_df.head()
test_df.head()
rul_df.head()
example_slice = train_df[(train_df.dataset_id == 'FD001') & (train_df.unit_id == 1)]



fig, axes = plt.subplots(7, 3, figsize=(15, 10), sharex=True)



for index, ax in enumerate(axes.ravel()):

    sensor_col = sensor_columns[index]

    example_slice.plot(x='cycle',y=sensor_col, ax=ax, color='black')

    

    if index % 3 == 0:

        ax.set_ylabel("Sensor Value", size=10)

    else:

        ax.set_ylabel("")

    

    ax.set_xlabel("Time (Cycles)")

    ax.set_title(sensor_col.title(), size=14)

    ax.legend_.remove()

    

fig.suptitle("Sensor Traces : Unit 1, Dataset 1", size=20, y=1.025)

fig.tight_layout()
all_units = train_df[train_df['dataset_id'] == 'FD001']['unit_id'].unique()

units_to_plot = np.random.choice(all_units, size=10, replace=False)

plot_data = train_df[(train_df['dataset_id'] == 'FD001') & (train_df['unit_id'].isin(units_to_plot))].copy()

plot_data.head()
for index, ax in enumerate(axes.ravel()):

    sensor_col = sensor_columns[index]

    for unit_id, group in plot_data.groupby('unit_id'):

        c = group.drop(columns=['dataset_id'],axis=1)
fig, axes = plt.subplots(7, 3, figsize=(15, 10), sharex=True)

for index, ax in enumerate(axes.ravel()):

    sensor_col = sensor_columns[index]

    for unit_id, group in plot_data.groupby('unit_id'):

        temp = group.drop(['dataset_id'],axis=1)

        (temp.plot(x='cycle', y=sensor_col, alpha=0.45, ax=ax, color='gray', legend=False))

        (temp.rolling(window=10, on='cycle').mean().plot(x='cycle', y=sensor_col, alpha=.75, ax=ax, color='black', legend=False));

    if index % 3 == 0:

        ax.set_ylabel('Sensor Value', size=10)

    else:

        ax.set_ylabel('')

    ax.set_title(sensor_col.title())

    ax.set_xlabel('Time (Cycles)')

fig.suptitle('All Sensor Traces: Dataset 1 (Random Sample of 10 Units)', size=20, y=1.025)

fig.tight_layout()
def cycles_until_failure(r, lifetimes):

    return r['cycle'] - lifetimes.ix[(r['dataset_id'], r['unit_id'])]
lifetimes = train_df.groupby(['dataset_id','unit_id'])['cycle'].max()

plot_data['ctf'] = plot_data.apply(lambda r: cycles_until_failure(r, lifetimes), axis=1)



fig, axes = plt.subplots(7,3, figsize=(15,10), sharex = True)

for index, ax in enumerate(axes.ravel()):

    sensor_col = sensor_columns[index]

    for unit_id, group in plot_data.groupby('unit_id'):

        temp = group.drop(['dataset_id'],axis=1)

        (temp.plot(x='ctf', y=sensor_col, alpha=0.45, ax=ax, color='gray', legend=False))

        (temp.rolling(window=10,on='ctf').mean().plot(x='ctf',y=sensor_col, alpha=.75, ax=ax, color='black',legend=False))

    if index % 3 == 0:

        ax.set_ylabel("Sensor Value", size=10)

    else:

        ax.set_ylabel("")

    ax.set_title(sensor_col.title())

    ax.set_xlabel('Time Before Failure (Cycles)')

    ax.axvline(x=0, color='r', linewidth=3)

    ax.set_xlim([None,10])

fig.suptitle("All Sensor Traces: Dataset 1 (Random Sample of 10 Units)", size=20, y=1.025)

fig.tight_layout()