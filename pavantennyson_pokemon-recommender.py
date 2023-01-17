import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity



import warnings

warnings.filterwarnings("ignore")



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



import ipywidgets as widgets

from ipywidgets import interact, interact_manual
poke_df = pd.read_csv('../input/pokemon/Pokemon.csv')

poke_df.head(10)
poke_df.shape
poke_df.isnull().sum()
poke_df['Type 2'].fillna(value='None',inplace=True)

poke_df['Pokemon Type'] = poke_df[['Name','Legendary']].apply(lambda x : 'Legendary' if x[1]==True else('Mega' if x[0].rfind('Mega ')!=-1 else 'Normal'),axis=1)



poke_df.drop(columns=['#','Legendary'],inplace=True)

poke_df.head()
@interact

def count_plot(Feature = ['Type 1','Type 2','Generation','Pokemon Type'],

               Hue = [None,'Type 1','Type 2','Generation','Pokemon Type'],

               Palette=plt.colormaps(),

               Style=plt.style.available, Width = (10,25,1), Height = (5,10,1), xTicks=(0,90,1)):

    

    

    plt.figure(figsize=(Width,Height))

    plt.style.use(Style)

    sns.countplot(x = Feature,

                  data = poke_df,

                  hue = Hue,

                  palette=Palette)

    plt.xticks(rotation=xTicks)
type_df = pd.DataFrame(np.zeros((poke_df.shape[0],len(poke_df['Type 2'].unique())),dtype=int),

                      index = poke_df.index,columns = sorted(poke_df['Type 2'].unique().tolist()))



for i in range(len(type_df)):

    types = []

    types.append(poke_df.loc[i,'Type 1'])

    types.append(poke_df.loc[i,'Type 2'])

    type_df.loc[i,types] = 1



type_df.head()
print(sorted(poke_df['Type 1'].unique().tolist()))

print(sorted(poke_df['Type 2'].unique().tolist()))
scaled_df = scaler.fit_transform(poke_df.drop(columns=['Name', 'Type 1', 'Type 2', 'Total', 'Generation', 'Pokemon Type']))

scaled_df = pd.DataFrame(scaled_df,columns=['HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed'])

scaled_df.head()
new_poke_df = pd.concat([type_df.drop(columns=['None']),pd.get_dummies(poke_df['Pokemon Type']),scaled_df],axis=1)

new_poke_df.head()
cos_sim = cosine_similarity(new_poke_df.values,new_poke_df.values)
cos_sim.shape
poke_index = pd.Series(poke_df.index,index=poke_df['Name'])

poke_index['Venusaur']
def recommend(pokemon,recommendations=5):

    index = poke_index[pokemon]

    similarity_score = list(enumerate(cos_sim[index]))

    sorted_score = sorted(similarity_score,key=lambda x : x[1],reverse=True)

    similar_pokemon = sorted_score[1:recommendations+1]

    poke_indices = [i[0] for i in similar_pokemon]

    return poke_df.iloc[poke_indices]
recommend('Charizard')
def rec_pokemon_byFilter(pokemon,

                         recommendations = 10,

                         include_original = False,

                         Type = None,

                         Type2 = None,

                         Generation = None,

                         pokemon_type = None):

    

    '''

    Recommends top 10 Pokemon which are similar to the given Pokemon

    

    By default number of recommendations is set to 10

    

    pokemon          : Name of the pokemon in string format

    recommendation   : Number of similar pokemon in the output, value must be Integer

    include_original : Includes the given pokemon in the output dataframe, value must be boolean

    Type             : Filter output by primary type

    Type2            : Filter output by secondary type

    Generation       : Filter output by Generation 

    pokemon_type     : Filter output by Pokemon Type 

    

    '''

    index = poke_index[pokemon]

    similarity_score = list(enumerate(cos_sim[index]))

    sorted_score = sorted(similarity_score,key=lambda x : x[1],reverse=True)

    

    

    if include_original == False:

        similar_pokemon = sorted_score[1:]

    elif include_original == True:

        similar_pokemon = sorted_score

    

    poke_indices = [i[0] for i in similar_pokemon]

    df = poke_df.iloc[poke_indices]

    

    if Type != None:

        df = df[(df['Type 1'] == Type)|(df['Type 2'] == Type)]

    else:

        pass

    

    if Type2 != None:

        df = df[(df['Type 1'] == Type2)|(df['Type 2'] == Type2)]

    else:

        pass

    

    if Generation != None:

        df = df[df['Generation'] == Generation]

    else:

        pass

    

    if pokemon_type != None:

        df = df[df['Pokemon Type'] == pokemon_type]

    else:

        pass

    

    

    return df.head(recommendations) if include_original == False else df.head(recommendations+1)
rec_pokemon_byFilter('Dragonite',recommendations=5)
rec_pokemon_byFilter('Dragonite',recommendations=10,pokemon_type='Normal',include_original=True)
rec_pokemon_byFilter('Dragonite',recommendations=10,include_original=True,Type='Dragon')
rec_pokemon_byFilter('Dragonite',recommendations=10,include_original=True,Type='Fire')