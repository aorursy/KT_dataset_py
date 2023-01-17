import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # accessing directory structure

import matplotlib.pyplot as plt # plotting

import seaborn as sns
haberman = pd.read_csv('../input/haberman.csv/haberman.csv')
haberman.head()
haberman.isnull().sum()
haberman['status'] = haberman.status.replace([1,2], ['survived','dead'])
g = sns.pairplot(haberman, hue = 'status')

g.map_diag(sns.distplot)

g.map_offdiag(plt.scatter)

g.add_legend()

g.fig.suptitle('FacetGrid plot', fontsize = 20)

g.fig.subplots_adjust(top=0.9);
gg = sns.boxplot(x='status',y='nodes', data=haberman)

gg.set_yscale('log')
age_corr = haberman

age_corr_dead = age_corr[age_corr['status'] == 'dead'].groupby(['age']).size().reset_index(name='count')

age_corr_dead.corr()
sns.regplot(x = 'age', y = 'count', data = age_corr_dead).set_title("Age vs Death count")
age_corr_survived = age_corr[age_corr['status'] == 'survived'].groupby(['age']).size().reset_index(name='count')

age_corr_survived.corr()
sns.regplot(x = 'age', y = 'count', data = age_corr_survived).set_title('Age vs Survived count')
year_corr = haberman

year_corr_dead = year_corr[year_corr['status'] == 'dead'].groupby(['year']).size().reset_index(name='count')

year_corr_dead.corr()
sns.regplot(x = 'year', y = 'count', data = year_corr_dead).set_title('Year vs death count')
year_corr_survived = year_corr[year_corr['status'] == 'survived'].groupby(['year']).size().reset_index(name='count')

year_corr_survived.corr()
sns.regplot(x = 'year', y = 'count', data = year_corr_survived).set_title('Year vs Survived count')
node_corr = haberman

node_corr_dead = node_corr[node_corr['status'] == 'dead'].groupby(['nodes']).size().reset_index(name = 'count')

node_corr_dead.corr()
sns.regplot(x = 'nodes', y = 'count', data = node_corr_dead).set_title('No of positive axillary nodes vs Death count')
node_corr_survived = node_corr[node_corr['status'] == 'survived'].groupby(['nodes']).size().reset_index(name ='count')

node_corr_survived.corr()
sns.regplot(x = 'nodes', y = 'count', data =node_corr_survived).set_title('No of positive axillary nodes vs Survived patients')