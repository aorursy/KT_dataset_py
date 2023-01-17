# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns



%matplotlib inline

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
parks = pd.read_csv("../input/park-biodiversity/parks.csv")

species = pd.read_csv("../input/park-biodiversity/species.csv")

species.head()
sns.heatmap(species.isnull())
df = pd.crosstab(species['Category'], species['Conservation Status'])

df 
style.use('default')

df.plot.barh(stacked=True, figsize=[15,8],

             edgecolor='white',

             width=1, 

             colormap='viridis_r')
mammal = species[species['Category']=='Mammal']

mammal_df = pd.crosstab(mammal['Order'], mammal['Conservation Status'], margins=True)

mammal_df.head()
mammal_df = mammal_df.drop(['All', 'Species of Concern'], axis=1)

mammal_df = mammal_df.drop(['All'], axis=0)
style.use('default')

mammal_df.plot.barh(stacked=True, 

                    figsize=[20,5], 

                    colormap='viridis', 

                    edgecolor='white', 

                    width=0.9,

                    title='Conservation Status of Mammals')
carn = mammal[mammal['Order']=='Carnivora']

ceta = mammal[mammal['Order']=='Cetacea']

arti = mammal[mammal['Order']=='Artiodactyla']

chir = mammal[mammal['Order']=='Chiroptera']



carn = carn[carn['Conservation Status']=='Endangered']

ceta = ceta[ceta['Conservation Status']=='Endangered']

arti = arti[arti['Conservation Status']=='Endangered']

chir = chir[chir['Conservation Status']=='Endangered']



carn = pd.crosstab(carn['Family'], carn['Conservation Status'], margins=True)

ceta = pd.crosstab(ceta['Family'], ceta['Conservation Status'], margins=True)

arti = pd.crosstab(arti['Family'], arti['Conservation Status'], margins=True)

chir = pd.crosstab(chir['Family'], chir['Conservation Status'], margins=True)



carn = carn.drop(['All'],axis=1)

carn = carn.drop(['All'], axis=0)

carn = carn.reset_index()

carn['specie'] = 'Carnivora'



ceta = ceta.drop(['All'],axis=1)

ceta = ceta.drop(['All'], axis=0)

ceta = ceta.reset_index()

ceta['specie'] = 'Cetacean'



arti = arti.drop(['All'],axis=1)

arti = arti.drop(['All'], axis=0)

arti = arti.reset_index()

arti['specie'] = 'Artiodactyla'



chir = chir.drop(['All'],axis=1)

chir = chir.drop(['All'], axis=0)

chir = chir.reset_index()

chir['specie'] = 'Chiroptera'
df_specie = pd.concat([ceta, carn, arti, chir],axis=0)
df_specie = pd.DataFrame(pd.concat([df_specie['Family'],

                       df_specie['Endangered'],

                       df_specie['specie']], axis=1))

df_specie
style.use('default')

plt.figure(figsize=[10,4.5])

sns.barplot(x = df_specie['specie'], 

            y = df_specie['Endangered'], 

            hue=df_specie['Family'],

            palette='viridis',

           dodge=True,

           edgecolor='None')

plt.xlabel('Specie')

plt.ylabel('Count')

plt.title('Endangered Mammals', fontsize=15)



plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
new_df = species[species['Category']=='Mammal']

new_df = new_df[['Category','Family','Abundance','Nativeness', 'Occurrence', 'Seasonality']]
df_ab = pd.crosstab(new_df['Family'],

           new_df['Abundance'])



df_nat = pd.crosstab(new_df['Family'],

           new_df['Nativeness'])



df_occ = pd.crosstab(new_df['Family'],

           new_df['Occurrence'])





df_seo = pd.crosstab(new_df['Family'],

           new_df['Seasonality'])
df_add = pd.concat([df_ab, df, df_nat, df_occ, df_seo], axis=1)

df_add
df_add.fillna(0, inplace=True)
sns.heatmap(df_add.corr())