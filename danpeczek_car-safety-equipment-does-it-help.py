import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



users = pd.read_csv('../input/accidents-in-france-from-2005-to-2016/users.csv', encoding='latin1')
users.head()
users.columns[users.isna().sum() != 0]
nans = users.isna()['secu'].sum() / users.shape[0]

nans
users = users.dropna(subset = ['secu'], axis='rows')
users['grav'] = users['grav'].astype(str) 

users['grav'] = users['grav'].replace({'1': 'unscathed', '2': 'killed', '3': 'hospitalized', '4': 'light injury'})

users['grav'] = pd.Categorical(users['grav'], ['unscathed', 'light injury', 'hospitalized', 'killed'])
# Split Secu to two columns - type and if equipment was used and renaming them

users.loc[:, 'equipment_used'] = users.loc[:, 'secu'].values % 10
ax = sns.countplot(x = 'grav', hue = 'equipment_used', data=users)

ax.set_title('Severity of accident based on using safety equipment')

ax.set_ylabel('Number of people')

plt.show()
users['equipment_used'] = users['equipment_used'].replace(0,1)

users['equipment_used'] = users['equipment_used'].astype(int)

users['equipment_used'] = users['equipment_used'].astype(str) 

users['equipment_used'] = users['equipment_used'].replace({'1': 'yes', '2': 'no', '3': 'unknown'})
users.loc[:, 'secu'] = users.loc[:, 'secu'].values // 10

users['secu'] = users['secu'].replace(0,1)

users.head()
plt.clf()

ax = sns.countplot(x = 'grav', hue = 'equipment_used', data=users.sort_values(by='grav'))

ax.set_title('Severity of accident based on using safety equipment')

ax.set_ylabel('Number of people')

plt.show()
plt.clf()

plt.figure(figsize=(10,7))

ax = sns.countplot(x = 'grav', hue = 'equipment_used', data=users[(users['catu'] == 1) | (users['catu'] == 2)])

ax.set_title('Severity of accident for drivers and passengers based on used safe equipment')

ax.set_ylabel('Number of people')

plt.show()