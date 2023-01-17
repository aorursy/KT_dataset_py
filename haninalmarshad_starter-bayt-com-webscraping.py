import pandas as pd

import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap
colors = ['powderblue', 'wheat', 'lightpink', 'lightcoral', 'rosybrown', 'darkseagreen',

          'beige',  'lightsalmon', 'palevioletred','peachpuff','powderblue', 'wheat', 'lightpink', 'lightcoral']

# Set custom color palette

c_palette = sns.set_palette(sns.color_palette(colors))
data = pd.read_csv('../input/jobs_bayt_c2.csv')
#data
data.info()
plt.figure(figsize=(16, 5))

chart =  sns.countplot(x='Job Role', data=data, palette=c_palette)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right');
plt.figure(figsize=(16, 5))

sns.countplot(x='Company Type', data=data.loc[data['Company Type']!='Unspecified'], palette=c_palette);
plt.figure(figsize=(16, 5))

sns.countplot(x='Company Type', data=data, palette=c_palette);
plt.figure(figsize=(16, 5))

sns.countplot(x='Employment Type', data=data,  palette=c_palette);
plt.figure(figsize=(16, 5))

sns.countplot(x='Career Level', data=data,  palette=c_palette);
plt.figure(figsize=(16, 5))

chart = sns.countplot(x='Job City', data=data,  palette=c_palette ,

                     order=pd.value_counts(data['Job City']).iloc[:12].index);
plt.figure(figsize=(16, 5))

chart = sns.countplot(x='Job City', data=data,  palette=c_palette)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right');
groupby_Dates = data.groupby('Date Posted').count()



plt.figure(figsize=(16, 10))

Date_chart = sns.countplot(y='Date Posted', data=data,  palette=c_palette ,

                     order=pd.value_counts(data['Date Posted']).iloc[:10].index);


corrs = data.drop(labels ='Job ID',axis=1).corr()

plt.figure(figsize=(10, 5))

mask = np.zeros_like(corrs, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns_palette = sns.color_palette(sns.diverging_palette( 220,10 ,sep=80, n=7)).as_hex()

ax = sns.heatmap(corrs, mask=mask, annot=True, cmap =ListedColormap(sns_palette))

ax.set_title('dataset correlation ')

# fix for mpl bug that cuts off top/bottom of seaborn viz

ax.set_ylim(len(corrs), -0.5)

plt.show()