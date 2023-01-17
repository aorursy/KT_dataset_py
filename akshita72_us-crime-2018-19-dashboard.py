# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/usa-crime-data-20182019/Crime_Data_USA.csv')

data.head()
data.shape
import matplotlib.pyplot as plt



# Missing Values

per = (data.isnull().sum()/data.shape[0])*100



print(per)



per.plot.barh()

plt.title('Missing Values')

plt.xlabel('Percentage')

plt.show()
data.dropna(inplace = True)

data.shape
data.dtypes
data['Year'] = data['Year'].astype('int')

data['Violent Crime'] = data['Violent Crime'].astype('int')

data['Murder'] = data['Murder'].astype('int')

data['Rape'] = data['Rape'].astype('int')

data['Robbery'] = data['Robbery'].astype('int')

data['Aggravated assault'] = data['Aggravated assault'].astype('int')

data['Property Crime'] = data['Property Crime'].astype('int')

data['Burglary'] = data['Burglary'].astype('int')

data['Larceny Theft'] = data['Larceny Theft'].astype('int')

data['Motor Vehicle Theft'] = data['Motor Vehicle Theft'].astype('int')

data['Arson'] = data['Arson'].astype('int')

data.dtypes
import seaborn as sns
fig_dims = (16, 12)

fig, ax = plt.subplots(5,2,figsize=fig_dims)



g1=sns.barplot(x='State', y='Violent Crime', hue='Year', ax=ax[0,0], data=data, ci=None)

g1.legend_.remove()

g1.set(xlabel='',xticklabels=[])



g2=sns.barplot(x='State', y='Murder', hue='Year', ax=ax[0,1], data=data, ci=None)

g2.legend_.remove()

g2.set(xlabel='',xticklabels=[])



g3=sns.barplot(x='State', y='Rape', hue='Year', ax=ax[1,0], data=data, ci=None)

g3.legend_.remove()

g3.set(xlabel='',xticklabels=[])



g4=sns.barplot(x='State', y='Robbery', hue='Year', ax=ax[1,1], data=data, ci=None)

g4.legend_.remove()

g4.set(xlabel='',xticklabels=[])



g5=sns.barplot(x='State', y='Aggravated assault', hue='Year', ax=ax[2,0], data=data, ci=None)

g5.legend_.remove()

g5.set(xlabel='',xticklabels=[])



g6=sns.barplot(x='State', y='Property Crime', hue='Year', ax=ax[2,1], data=data, ci=None)

g6.legend_.remove()

g6.set(xlabel='',xticklabels=[])



g7=sns.barplot(x='State', y='Burglary', hue='Year', ax=ax[3,0], data=data, ci=None)

g7.legend_.remove()

g7.set(xlabel='',xticklabels=[])



g8=sns.barplot(x='State', y='Larceny Theft', hue='Year', ax=ax[3,1], data=data, ci=None)

g8.legend_.remove()

g8.set(xlabel='',xticklabels=[])



g9=sns.barplot(x='State', y='Motor Vehicle Theft', hue='Year', ax=ax[4,0], data=data, ci=None)

g9.legend_.remove()

g9.set_xticklabels(g9.get_xticklabels(),rotation=85)



g10=sns.barplot(x='State', y='Arson', hue='Year', ax=ax[4,1], data=data, ci=None)

g10.legend_.remove()

g10.set_xticklabels(g10.get_xticklabels(),rotation=85)



handles, labels = g10.get_legend_handles_labels()

fig.legend(handles, labels, title='Year', loc='upper center')

plt.show()
new_york = data[data['State']=='NEW YORK'].drop(['Year','Population'],axis=1)

fig_dims = (14, 10)

fig, ax = plt.subplots(figsize=fig_dims)

sns.barplot(ax=ax, data=new_york, ci=None)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

ax.set_xlabel('Crimes Type')

ax.set_ylabel('Number of Crimes')

plt.show()
ax = sns.catplot(data=data[data['State'].isin(['CALIFORNIA','NEW YORK'])].drop(['Year','Population'],axis=1),

             ci=None, col='State')

labels = ['Violent Crime','Murder','Rape','Robbery','Aggravated assault',

          'Property Crime','Burglary','Larceny Theft','Motor Vehicle Theft','Arson']

ax.set_xticklabels(labels,rotation=65)

plt.show()
xx18 = data[data['Year']==2018].drop(['City','Year','Population'],axis=1).groupby('State').sum()

xx19 = data[data['Year']==2019].drop(['City','Year','Population'],axis=1).groupby('State').sum()
fig_dims = (16, 12)

fig, ax = plt.subplots(1,2,figsize=fig_dims)



gg1 = sns.heatmap(

    data=xx18, 

    square=True, # make cells square

    cbar_kws={'fraction' : 0.1}, # shrink colour bar

    cmap='OrRd', # use orange/red colour map

    linewidth=1, # space between cells

    ax=ax[0]

)

gg1.set(title='2018')



gg2 = sns.heatmap(

    data=xx19, 

    square=True, # make cells square

    cbar_kws={'fraction' : 0.1}, # shrink colour bar

    cmap='OrRd', # use orange/red colour map

    linewidth=1, # space between cells

    ax=ax[1]

)

gg2.set(title='2019')



plt.show()