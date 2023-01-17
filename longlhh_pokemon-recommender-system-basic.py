import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
pokemon = pd.read_csv('../input/pokemon/Pokemon.csv')
pokemon.head()
pokemon.info()
Legend = pd.get_dummies(pokemon['Legendary'],drop_first=True)
pokemon.drop(['#','Legendary'],axis=1,inplace=True)

pokemon = pd.concat([pokemon,Legend],axis=1)

pokemon.rename({True:'Legend'},axis=1, inplace=True)
pokemon.head()
def type_numbering(string) : 

    if string == 'Normal' :

        return 1

    elif string== 'Fire' :

        return 2

    elif string == 'Fighting' :

        return 3

    elif string == 'Water' :

        return 4

    elif string == 'Flying' :

        return 5

    elif string == 'Grass' :

        return 6

    elif string == 'Poison' :

        return 7

    elif string == 'Electric' :

        return 8

    elif string == 'Ground' :

        return 9

    elif string == 'Psychic' :

        return 10

    elif string == 'Rock' :

        return 11

    elif string == 'Ice' :

        return 12

    elif string == 'Bug' :

        return 13

    elif string == 'Dragon' :

        return 14

    elif string == 'Ghost' :

        return 15

    elif string == 'Dark' :

        return 16

    elif string == 'Steel' :

        return 17

    elif string == 'Fairy' :

        return 18

    else :

        return 0
pokemon['Type 1'] = pokemon['Type 1'].apply(type_numbering)

pokemon['Type 2'] = pokemon['Type 2'].apply(type_numbering)
pokemon.head()
indices = pd.Series(pokemon.index, index=pokemon['Name'])
pokematrix = pokemon.drop('Name',axis=1)
def recommendation(pkm):

    idx = indices[pkm]

    sim_scores = []

    for i in range(pokemon.shape[0]):

        sim_scores.append(np.linalg.norm(pokematrix.loc[idx]-pokematrix.loc[i]))

    sim_scores = list(enumerate(sim_scores))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=False)

    sim_scores = sim_scores[1:31]

    pkm_indices = [i[0] for i in sim_scores]

    sim_pkm = pokemon.iloc[pkm_indices].head(10)

    return sim_pkm
recommendation('Pikachu')
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(pokematrix,pokematrix)
def recommendation_2(pkm):

    idx = indices[pkm]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    pkm_indices = [i[0] for i in sim_scores]    

    sim_pkm = pokemon.iloc[pkm_indices].head(10)

    return sim_pkm
recommendation_2('Pikachu')
def check_type(x,a,b):

    pkm_type_1 = x['Type 1']

    pkm_type_2 = x['Type 2']

    if (pkm_type_1 == a) and (pkm_type_2 == b):

        return 1

    elif (pkm_type_1 == a) or (pkm_type_2 == b):

        return 0.5

    else:

        return 0
def enhanced_recommendation(pkm):

    idx = indices[pkm]

    pkm_type1= pokematrix.loc[idx]['Type 1']

    pkm_type2= pokematrix.loc[idx]['Type 2']

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    pkm_indices = [i[0] for i in sim_scores]

    

    sim_pkm = pokemon.iloc[pkm_indices].copy()

    sim_pkm['sim_type'] = sim_pkm.apply(lambda x: check_type(x,pkm_type1,pkm_type2), axis=1)

    sim_pkm = sim_pkm.sort_values('sim_type', ascending=False).head(10)

    return sim_pkm
enhanced_recommendation('Pikachu')
enhanced_recommendation('Charizard')
enhanced_recommendation('Bulbasaur')