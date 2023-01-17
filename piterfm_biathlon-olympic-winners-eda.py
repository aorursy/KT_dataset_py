# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for plotting
import matplotlib.pyplot as plt # for plotting

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/olympic_biathlon.csv')
print('Biathlon data: \nRows: {}\nCols: {}'.format(data.shape[0], data.shape[1]))
data.head()
list(data.columns)
data.info()
data.describe(include='all')
data[data['Country'].isna()]
data['Country'] = np.where(data['Country'].isna(), data['Winner'], data['Country'])
data.info()
data['Game URL'] = 'https://www.olympic.org' + data['Game URL']
data.head()
data.drop(['Discipline URL','Game URL'], inplace=True, axis=1)
data.head()
data['Discipline'].value_counts()
disciplines = {'4x7.5km Relay': 'Relay', 
               '4x6km Relay':'Relay', 
               '2x6km Women + 2x7.5km Men Relay':'Relay', 
               '3x7.5km Relay':'Relay',
               '20km Individual':'Individual',
               '15km Individual':'Individual',
               '10km Sprint':'Sprint',
               '7.5km Sprint':'Sprint',
               '12.5km Pursuit':'Pursuit',
               '10km Pursuit':'Pursuit',
               '15km Mass Start':'Mass Start',
               '12.5km Mass Start':'Mass Start'
              }
data['Discipline'] = data['Discipline'].map(disciplines)
data['Discipline'].value_counts()
sns.set(style='whitegrid')
country = data['Country'].value_counts()[:]
plt.figure(figsize=(12,12))
ax = sns.barplot(x=country.values, y=country.index)
plt.title('All Olympic Winners in Biathlon by Country', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Country', fontsize=18)

for patch, value in zip(ax.patches, country.values[:]):
#   print(patch.get_y(), ' ' ,patch.get_height(), ' ', patch.get_x(), ' ', patch.get_width(), ' ', value)
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 
sns.set(style='whitegrid')
plt.figure(figsize=(12,12))
ax = sns.countplot(y='Country',data=data[data['Gender']=='Men'], order = data[data['Gender']=='Men']['Country'].value_counts().index)
plt.title('All Men Olympic Winners in Biathlon by Country', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Country', fontsize=18)

for patch, value in zip(ax.patches, data[data['Gender']=='Men']['Country'].value_counts()):
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 
sns.set(style='whitegrid')
plt.figure(figsize=(12,12))
ax = sns.countplot(y='Country',data=data[data['Gender']=='Women'], order = data[data['Gender']=='Women']['Country'].value_counts().index)
plt.title('All Women Olympic Winners in Biathlon by Country', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Country', fontsize=18)

for patch, value in zip(ax.patches, data[data['Gender']=='Women']['Country'].value_counts().values[:]):
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 
data[(data['Discipline']!='Relay')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')]['Winner'].value_counts().values>=3]
sns.set(style='whitegrid')
biathletes = data[(data['Discipline']!='Relay')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')]['Winner'].value_counts().values>=3]
plt.figure(figsize=(12,12))
ax = sns.barplot(x=biathletes.values, y=biathletes.index)
plt.title('All Olympic Winners in Biathlon by Athlete', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Athlete', fontsize=18)

for patch, value in zip(ax.patches, biathletes.values[:]):
#   print(patch.get_y(), ' ' ,patch.get_height(), ' ', patch.get_x(), ' ', patch.get_width(), ' ', value)
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 
data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts().values>=2]
sns.set(style='whitegrid')
biathletes = data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts()[data[(data['Discipline']!='Relay')&(data['Medal']=='G')]['Winner'].value_counts().values>=2]
plt.figure(figsize=(12,12))
ax = sns.barplot(x=biathletes.values, y=biathletes.index)
plt.title('All GOLD Olympic Winners in Biathlon by Athlete', fontsize=18)
plt.xlabel('Number of Medals', fontsize=18)
plt.ylabel('Athlete', fontsize=18)

for patch, value in zip(ax.patches, biathletes.values[:]):
#   print(patch.get_y(), ' ' ,patch.get_height(), ' ', patch.get_x(), ' ', patch.get_width(), ' ', value)
  ax.text(patch.get_width()+0.5,
          patch.get_y()+0.65,
          value,
          ha="left",
          fontsize=16) 
grouped_data = data.groupby(['Country','Discipline'])['Gender'].count().sort_index().reset_index()
grouped_data
grouped_data = grouped_data.pivot('Country','Discipline','Gender')
grouped_data
plt.figure(figsize=(10,10))
sns.heatmap(grouped_data, cmap='RdYlGn', annot=True, linewidths=.5)
sns.set(style='whitegrid')
plt.figure(figsize=(8,8))
filter_germany=(data['Country']=='GER')&(data['Gender']=='Women')
ax = sns.countplot(x='Year', data=data[filter_germany])#, order = data[filter_germany]['Year'].value_counts().index)
plt.title('All Germany Women Olympic Winners in Biathlon by Year', fontsize=18)
plt.ylabel('Number of Medals', fontsize=18)
plt.xlabel('Women Germany Team', fontsize=18)
sns.set(style='whitegrid')
plt.figure(figsize=(8,8))
filter_norway=(data['Country']=='NOR')&(data['Gender']=='Men')
ax = sns.countplot(x='Year', data=data[filter_norway])
plt.title('All Norway Men Olympic Winners in Biathlon by Year', fontsize=18)
plt.ylabel('Number of Medals', fontsize=18)
plt.xlabel('Men Norway Team', fontsize=18)
