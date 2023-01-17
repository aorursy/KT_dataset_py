# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
data = pd.read_csv('../input/rainfall-data-india-since-1901/rainfall_India_2017.csv').rename(columns=str.lower)

data.head(3)
data.info()
print('Dataset comprises of {} observations and {} characteristics'.format(data.shape[0],data.shape[1]))

print('\nUnique Values: ',data.nunique())

print('\nMissing Values: ',data.isna().sum())
data.describe().T
len(data['year'].unique()), data['year'].min(), data['year'].max()
data = data.fillna(0)
data.hist(figsize=(14,10),color='orange');
plt.style.use('fivethirtyeight')

data.groupby('year')['annual'].mean().plot(title='Annual Rainfall Of The 117 Years ',

                                          figsize=(13,8),

                                          c='g',

                                          marker='s')

plt.ylabel('Rainfalls (mm)');
data.drop(['annual','subdivision'],axis=1).groupby('year').mean().T.plot(title='Monthly  Rainfall of The 117 Years',figsize=(12,7),alpha=.15,legend=False);
plt.style.use('seaborn-colorblind')

fig = plt.figure(figsize=(16, 8))

plt.xticks(rotation='vertical')

ax=sns.boxplot(x='subdivision', y='annual', data=data);

ax.axhline(data['annual'].mean(),linestyle='-.',linewidth=2,color='g');
data.groupby('subdivision')[['annual']].mean().sort_values('annual').head(3).plot.barh(title='Top Most Lowest Rainfalling Subdivision (3)',color='indigo',legend=False);
data.groupby('subdivision')[['annual']].mean().sort_values('annual').tail(3).plot.barh(title='Top Most Highest Rainfalling Subdivision (3)',color='gold',legend=False);

li = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',

       'aug', 'sep', 'oct', 'nov', 'dec']

h = data.drop(li,axis=1).dropna().sort_values('annual').head(1).values.tolist()

l = data.drop(li,axis=1).dropna().sort_values('annual').tail(1).values.tolist()

print('\nThe highest {} millimeter rainfall happens in {} & the year of {}. '.format(l[0][2],l[0][0],l[0][1]))

print('\nThe lowest {} millimeter rainfall happens in {} & the year of {}. '.format(h[0][2],h[0][0],h[0][1]))
plt.style.use('seaborn-poster')

ax = data.drop(['annual','year'],axis=1).groupby('subdivision').sum().sort_values('jul').plot.bar(stacked=True,figsize=(16, 8),

                                                                               title='Monthly Rainfalls of All Subdivision')

ax.set_xlabel('Subdivision')

ax.set_ylabel('Rainfalls (mm)')

patches, labels = ax.get_legend_handles_labels()    

ax.legend(patches, labels, loc='upper left',ncol=6,borderaxespad=.1)

plt.show()
data.groupby('year')['annual'].sum().rolling(10).mean().plot(title='Rainfall of 10 Years Average Of The 117 Years',figsize=(14,6),linewidth=2,c='mediumvioletred',marker='*');
colors = ['aqua', 'black',  'brown', 'chartreuse', 'coral', 'cyan', 'darkblue','darkgreen', 'fuchsia','gold', 'goldenrod', 'green', 'grey',

          'indigo', 'khaki', 'lightblue', 'lightgreen', 'lime', 'magenta', 'maroon', 'navy', 'olive', 'orange','orangered', 'orchid', 'plum',

          'purple', 'salmon', 'sienna', 'silver', 'tan', 'teal', 'tomato', 'turquoise', 'violet', 'wheat', 'yellow','yellowgreen']
# hidding output

rfs = sorted(set(data["subdivision"].to_list()))

plt.style.use('seaborn')

plt.style.use('seaborn-pastel')



for sub in rfs:

    fig, ax = plt.subplots(figsize=(14,2))

    rainfalls= data[data.subdivision == sub]

    ax.bar(rainfalls['year'],rainfalls['annual'],color=random.choice(colors),

                 edgecolor='wheat',label='Rainfall')

    

    ax.axhline(rainfalls['annual'].mean(),linestyle='-.',linewidth=2,color='b')

    ax.set(title=' Annual Rainfall of '+sub+' Subdivision',xlabel='Year',ylabel='Rainfall (mm)',)

    patches, labels = ax.get_legend_handles_labels()

    ax.legend(patches, labels, loc='upper left',bbox_to_anchor=(0.05, 0., 0.5, 0.3),borderaxespad=.3)

    plt.show()
# hidding output

rfs = sorted(set(data["subdivision"].to_list()))

td = data.drop(['annual','year'],axis=1)

plt.style.use('seaborn-pastel')

for sub in rfs:



    rainfalls=td[td.subdivision==sub]

    ax = rainfalls.plot.bar(title='Monthly Rainfall of '+sub+' Subdivision', fontsize=13, figsize=(27,5), stacked=True)

    ax.set_xticklabels(data['year'],rotation=90);

    patches, labels = ax.get_legend_handles_labels()

    ax.legend(patches, labels, loc='lower left',ncol=1,bbox_to_anchor=(1,0.17),borderaxespad=.4)

    ax.title.set_size(18)

    ax.set_xlabel('Year',size=16)

    ax.set_ylabel('Rainfall (mm)',size=16)

    plt.show()

plt.matshow(data.drop(['annual','subdivision','year'], axis=1).corr())

plt.grid(False);