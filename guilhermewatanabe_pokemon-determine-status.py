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
pokemon_csv=pd.read_csv('../input/complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv', index_col=0)

#print(pokemon_csv.head())

#print(pokemon_csv.info())
#Let's clean the data

pokemon_w_against=pokemon_csv.drop(['german_name','japanese_name'], axis=1)

pokemon=pokemon_w_against.drop(pokemon_w_against.columns[list(range(30,48))], axis=1)

#print(pokemon.head())

#print(pokemon.info())
#Let's fill the catch rates

#print(pokemon.name[(pokemon.catch_rate.isnull())&(pokemon.name.str.contains('Galarian'))])

#pd.isnull(pokemon.iloc[69]['catch_rate'])



for i in range(pokemon.shape[0]):

    if pd.isnull(pokemon.iloc[i]['catch_rate']) and pokemon.iloc[i]['pokedex_number']==pokemon.iloc[i-1]['pokedex_number']:

        pokemon.at[i,'catch_rate']=pokemon.iloc[i-1]['catch_rate']

        pokemon.at[i,'base_friendship']=pokemon.iloc[i-1]['base_friendship']

        pokemon.at[i,'base_experience']=pokemon.iloc[i-1]['base_experience']

pokemon[pokemon.catch_rate.isnull()].name



#We realize that the galar pokemon do not have their catch rates updated...

#Let's check for Megas

substring='Mega '# I am looking at you Meganium

mega_pokemon=pokemon[pokemon.name.str.contains(substring)]

print(mega_pokemon.shape[0])



#We do have 48 megas in total, so it is good
for i in range(pokemon.shape[0]):

    if ('Mega 'in pokemon.iloc[i]['name']) and pokemon.iloc[i]['status']=='Normal':

        pokemon.at[i,'status']='Mega'

pokemon[pokemon['status']=='Mega'].shape[0]

#We do have 42 non-legendary Megas, so it is working good

#For good practice, let's create categorical and integer data types where it is convinient

pokemon=pokemon.astype({'generation':'category','status':'category', 'type_number':'category',

                        'type_1':'category','type_2':'category','abilities_number':'category',

                        'egg_type_number':'category','egg_type_1':'category','egg_type_2':'category'})

pokemon=pokemon.astype({'total_points':'int64','hp':'int64','attack':'int64',

                        'defense':'int64','sp_attack':'int64','sp_defense':'int64','speed':'int64'})

#pokemon.info()
category=pokemon.groupby('status')

category.size()

category['total_points'].mean()

category.head(3)