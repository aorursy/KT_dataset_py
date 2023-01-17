# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pokemon = pd.read_csv('../input/Pokemon.csv')

pokemon.head()
pokemon.drop('#',axis=1,inplace=True)

pokemon.head()
pokemon.info()
t1_count = pokemon['Type 1'].value_counts()

t2_count = pokemon['Type 2'].value_counts()
types = pd.concat([t1_count,t2_count],axis=1,sort=False)

types
types['Total'] = types['Type 1'] + types['Type 2']

types.sort_values(by='Total',ascending=False,inplace=True)

types
plt.figure(figsize=(12,6))

plt.bar(types.index,types['Total'])

plt.xticks(rotation=60)

plt.show()
plt.figure(figsize=(10,10))

plt.pie(types['Total'],labels=types.index,shadow=True,autopct='%1.1f%%',

        explode=(0.4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

plt.show()
pokemon['Generation'].value_counts()
pokemon['Legendary'].value_counts()
pokemon_num = pokemon.loc[:,'Total':'Speed']

sns.pairplot(pokemon_num)

plt.show()
sns.heatmap(pokemon_num.corr(),annot=True)

plt.show()
fig,ax = plt.subplots(7,2,figsize=(16,50))

sns.distplot(pokemon['Total'],ax=ax[0,0])

sns.boxplot(pokemon['Total'],ax=ax[0,1])

sns.distplot(pokemon['HP'],ax=ax[1,0])

sns.boxplot(pokemon['HP'],ax=ax[1,1])

sns.distplot(pokemon['Attack'],ax=ax[2,0])

sns.boxplot(pokemon['Attack'],ax=ax[2,1])

sns.distplot(pokemon['Defense'],ax=ax[3,0])

sns.boxplot(pokemon['Defense'],ax=ax[3,1])

sns.distplot(pokemon['Sp. Atk'],ax=ax[4,0])

sns.boxplot(pokemon['Sp. Atk'],ax=ax[4,1])

sns.distplot(pokemon['Sp. Def'],ax=ax[5,0])

sns.boxplot(pokemon['Sp. Def'],ax=ax[5,1])

sns.distplot(pokemon['Speed'],ax=ax[6,0])

sns.boxplot(pokemon['Speed'],ax=ax[6,1])

plt.show()
fig,ax = plt.subplots(7,1,figsize=(16,30))

sns.boxplot(x='Type 1',y='Total',data=pokemon,ax=ax[0])

sns.boxplot(x='Type 1',y='HP',data=pokemon,ax=ax[1])

sns.boxplot(x='Type 1',y='Attack',data=pokemon,ax=ax[2])

sns.boxplot(x='Type 1',y='Defense',data=pokemon,ax=ax[3])

sns.boxplot(x='Type 1',y='Sp. Atk',data=pokemon,ax=ax[4])

sns.boxplot(x='Type 1',y='Sp. Def',data=pokemon,ax=ax[5])

sns.boxplot(x='Type 1',y='Speed',data=pokemon,ax=ax[6])

plt.show()
fig,ax = plt.subplots(7,1,figsize=(12,30))

sns.boxplot(x='Generation',y='Total',data=pokemon,ax=ax[0])

sns.boxplot(x='Generation',y='HP',data=pokemon,ax=ax[1])

sns.boxplot(x='Generation',y='Attack',data=pokemon,ax=ax[2])

sns.boxplot(x='Generation',y='Defense',data=pokemon,ax=ax[3])

sns.boxplot(x='Generation',y='Sp. Atk',data=pokemon,ax=ax[4])

sns.boxplot(x='Generation',y='Sp. Def',data=pokemon,ax=ax[5])

sns.boxplot(x='Generation',y='Speed',data=pokemon,ax=ax[6])

plt.show()
fig,ax = plt.subplots(4,2,figsize=(16,30))

sns.boxplot(x='Legendary',y='Total',data=pokemon,ax=ax[0,0])

sns.boxplot(x='Legendary',y='HP',data=pokemon,ax=ax[0,1])

sns.boxplot(x='Legendary',y='Attack',data=pokemon,ax=ax[1,0])

sns.boxplot(x='Legendary',y='Defense',data=pokemon,ax=ax[1,1])

sns.boxplot(x='Legendary',y='Sp. Atk',data=pokemon,ax=ax[2,0])

sns.boxplot(x='Legendary',y='Sp. Def',data=pokemon,ax=ax[2,1])

sns.boxplot(x='Legendary',y='Speed',data=pokemon,ax=ax[3,0])

plt.show()
from scipy.stats import shapiro,levene,bartlett,ttest_1samp,wilcoxon,ttest_ind,mannwhitneyu,f_oneway

l_grp = pokemon.groupby('Legendary')

legendary_true = l_grp.get_group(True)

legendary_false = l_grp.get_group(False)
# check for normality

print(shapiro(legendary_true['Total']))

print(shapiro(legendary_false['Total']))
# comparing variances

bartlett(legendary_true['Total'],legendary_false['Total'])
mannwhitneyu(legendary_true['Total'], legendary_false['Total'])
gen_grp = pokemon.groupby('Generation')

gen1 = gen_grp.get_group(1)

gen2 = gen_grp.get_group(2)

gen3 = gen_grp.get_group(3)

gen4 = gen_grp.get_group(4)

gen5 = gen_grp.get_group(5)

gen6 = gen_grp.get_group(6)
f_oneway(gen1['Total'],gen2['Total'],gen3['Total'],gen4['Total'],gen5['Total'],gen6['Total'])
sns.countplot(x='Generation',hue='Legendary',data=pokemon)

plt.show()