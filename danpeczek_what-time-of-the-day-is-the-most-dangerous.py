import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



dataset = pd.read_csv('../input/accidents-in-france-from-2005-to-2016/caracteristics.csv', encoding='latin1')

dataset.head()
dataset.columns[dataset.isna().sum() != 0]
dataset['lum'] = dataset['lum'].astype(str) 

dataset['lum'] = dataset['lum'].replace({'2': 'Twilight or dawn', '1': 'Full day', '3': 'Night no public lighting', '4': 'Night lighting not lit', '5': 'Night with public lighting on'})
plt.clf()

ax = sns.countplot(y = 'lum', data=dataset)

ax.set_title('Number of accidents based on the lightning conditions')

ax.set_xlabel('Number of accidents')

ax.set_ylabel('Light conditions')

plt.show()
dataset.dtypes
dataset['hr'] = dataset['hrmn']/100

dataset['hr'] = dataset['hr'].astype('int64')

dataset.head()
plt.clf()

plt.figure(figsize=(10,7))

ax = sns.countplot(x='hr', data=dataset, palette="GnBu_d")

ax.set_title('Number of accidents during 24-hours')

ax.set_xlabel('Hour')

ax.set_ylabel('Number of accidents')

plt.show()