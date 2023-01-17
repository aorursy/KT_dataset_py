import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
pokemons = pd.read_csv('../input/pokemonGO.csv')

print(pokemons.head(3))
# Columns/Fields in the dataset

print(pokemons.columns.tolist())
# Find NaN in the dataset

print(pokemons[pokemons.isnull().any(axis=1)])
newdf = pd.get_dummies(pokemons,prefix=['Type1','Type2'],columns=['Type 1','Type 2'])

print(newdf.sample(2))
for col in newdf.columns:

    if 'Type' in col:

        newdf[[col]] = newdf[[col]].astype(int)
print(newdf.sample(2))
print(newdf.columns.tolist())
pokemons['IsBugType'] = newdf['Type1_Bug']

pokemons['IsDragonType'] = newdf['Type1_Dragon']

pokemons['IsElectricType'] = newdf['Type1_Electric']

pokemons['IsFairyType'] = (newdf['Type1_Fairy'] | newdf['Type2_Fairy'])

pokemons['IsFightingType'] = (newdf['Type1_Fighting'] | newdf['Type2_Fighting'])

pokemons['IsFlyingType'] = newdf['Type2_Flying']

pokemons['IsGhostType'] = newdf['Type1_Ghost']

pokemons['IsGrassType'] = (newdf['Type1_Grass'] | newdf['Type2_Grass'])

pokemons['IsGroundType'] = (newdf['Type1_Ground'] | newdf['Type2_Ground'])

pokemons['IsIceType'] = (newdf['Type1_Ice'] | newdf['Type2_Ice'])

pokemons['IsNormalType'] = newdf['Type1_Normal']

pokemons['IsPoisonType'] = (newdf['Type1_Poison'] | newdf['Type2_Poison'])

pokemons['IsPsychicType'] = (newdf['Type1_Psychic'] | newdf['Type2_Psychic'])

pokemons['IsRockType'] = (newdf['Type1_Rock'] | newdf['Type2_Rock'])

pokemons['IsSteelType'] = newdf['Type2_Steel']

pokemons['IsWaterType'] = (newdf['Type1_Water'] | newdf['Type2_Water'])
print(pokemons.sample(2))
print(pokemons.shape)

pokemons.drop(['Type 1','Type 2'],axis=1,inplace=True)

print(pokemons.shape)

print(pokemons.head(2))
typedf = pd.DataFrame(columns=['Type','Count'])

typedf.Count = typedf.Count.astype(int)

for col in pokemons.columns:

    if 'Is' in col:

        count = pokemons[col].sum()

        #print('Count of ',col,' Pokemon: ',str(count))

        typedf = typedf.append({'Type':col,'Count':count},ignore_index=True)
typedf.sort_values('Count',axis=0,ascending=False)
pokemons[['Pokemon No.','Name','Max CP']].sort_values('Max CP',axis=0,ascending=False).head(10)
pokemons[['Pokemon No.','Name','Max HP']].sort_values('Max HP',axis=0,ascending=False).head(10)