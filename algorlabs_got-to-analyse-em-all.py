# import all libs

from os import path

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline
# import dataset

df = pd.read_csv('../input/Pokemon.csv')
df.head()
# Top 10 most powerful pokemon

top10_df = df.sort_values(['Total'],ascending=False)[0:10]

plt.figure(figsize=(15,5))

sns.barplot(x='Total',y='Name',data=top10_df,estimator=sum)
# Top 10 most poweful excluding variations of pokemon

# change # column name to Pokeindex

df = df.rename(columns={'#':'Pokeindex'})

df.head()
# clean off duplicates

no_var_df = df.drop_duplicates(subset='Pokeindex')

#plot again

no_var_df = no_var_df.sort_values(['Total'],ascending=False)[0:10]

plt.figure(figsize=(15,5))

sns.barplot(x='Total',y='Name',data=no_var_df,estimator=sum)

    
# Most powerful type of pokemon based on type 1

plt.figure(figsize=(10,5))

sns.barplot(x='Total',y='Type 1',data=df)
# Which is the most powerful type of combination pokemon?

df['combi_type'] = df['Type 1'] +' & '+ df['Type 2']

df.head()

combi_df = df.dropna()

combi_df.head()
plt.figure(figsize=(20,35))

sns.barplot(x='Total',y='combi_type',data=combi_df)
# % of Pokemon type

plt.figure(figsize=(12,12))

count_type = combi_df['Type 1'].value_counts()

explode = np.zeros_like(np.array(count_type),dtype=float)

explode[1] = 0.1

g = count_type.plot.pie(autopct='%1.1f%%',shadow=True,explode=explode)

g.legend(loc='upper left',bbox_to_anchor=(1,1))
# slice and combine the frames



# slice for grass only

g_df = combi_df[combi_df['Type 1']=='Grass']

# slice for fire only

f_df= combi_df[combi_df['Type 1']=='Fire']

#add them both to create new dataframe

gnf_df = pd.concat([g_df,f_df])

gnf_df['Type 1'].value_counts()
gnf_df.drop(['Pokeindex','Total'],axis=1,inplace=True)

sns.pairplot(gnf_df,hue='Type 1')