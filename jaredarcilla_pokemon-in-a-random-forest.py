import numpy as np

import pandas as pd
combats = pd.read_csv('../input/pokemon-challenge/combats.csv')

poke = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
combats.head()
poke.head()
weaknesses = pd.read_json("../input/pokemon-weaknesses/types.json")

weaknesses.set_index('name',inplace=True)

weaknesses.head()
poke.set_index('#', inplace=True)
first_poke = combats.join(poke, on='First_pokemon',how='left')
second_poke = combats.join(poke, on='Second_pokemon', how = 'left')
first_poke.drop(['First_pokemon', 'Second_pokemon', 'Winner', 'Name', 'Generation', 'Legendary'], axis=1, inplace=True)

second_poke.drop(['First_pokemon', 'Second_pokemon', 'Winner', 'Name', 'Generation', 'Legendary'], axis=1, inplace=True)


winners = np.zeros(len(combats))

for i, row in combats.iterrows():

    if row[0] == row[2] :

        winners[i] = -1

    else:

        winners[i] = 1     
first_poke.head()
type_matchup = np.zeros(len(combats))

for i, row in combats.iterrows() :



    poke1type1 = poke.ix[int(row[0]), 'Type 1']

    poke1type2 = poke.ix[int(row[0]), 'Type 2']

    poke2type1 = poke.ix[int(row[1]), 'Type 1']

    poke2type2 = poke.ix[int(row[1]), 'Type 2']

    

    

    #is poke1 weak to poke 2

    if poke2type1 in weaknesses.ix[poke1type1, 'weaknesses']:

        type_matchup[i] += 1 #lower negative values mean poke 1 is at adv, higher positive values mean poke 2 is at adv

    elif poke2type1 in weaknesses.ix[poke1type1, 'immunes']:

        type_matchup[i] += 2

    

    if pd.notnull(poke1type2) and poke2type1 in weaknesses.ix[poke1type2, 'weaknesses']:

        type_matchup[i] += 1

    elif pd.notnull(poke1type2) and poke2type1 in weaknesses.ix[poke1type2, 'immunes']:

        type_matchup[i] += 2

    

    if pd.notnull(poke2type2) and poke2type2 in weaknesses.ix[poke1type1, 'weaknesses']:

        type_matchup[i] += 1 

    elif pd.notnull(poke2type2) and poke2type2 in weaknesses.ix[poke1type1, 'immunes']:

        type_matchup[i] += 2

    

    if pd.notnull(poke1type2) and pd.notnull(poke2type2) and poke2type2 in weaknesses.ix[poke1type2, 'weaknesses']:

        type_matchup[i] += 1 

    elif pd.notnull(poke1type2) and pd.notnull(poke2type2) and poke2type2 in weaknesses.ix[poke1type2, 'immunes']:

        type_matchup[i] += 2

    

    

    #is poke 2 weak to poke 1 

    if poke1type1 in weaknesses.ix[poke2type1, 'weaknesses']:

        type_matchup[i] -= 1

    elif poke1type1 in weaknesses.ix[poke2type1, 'immunes']:

        type_matchup[i] -= 2

    

    if pd.notnull(poke2type2) and poke1type1 in weaknesses.ix[poke2type2, 'weaknesses']:

        type_matchup[i] -= 1

    elif pd.notnull(poke2type2) and poke1type1 in weaknesses.ix[poke2type2, 'immunes']:

        type_matchup[i] -= 2

    

    if pd.notnull(poke1type2) and poke1type2 in weaknesses.ix[poke2type1, 'weaknesses']:

        type_matchup[i] -= 1 #lower negative values mean poke 1 is at adv, higher positive values mean poke 2 is at adv

    elif pd.notnull(poke1type2) and poke1type2 in weaknesses.ix[poke2type1, 'immunes']:

        type_matchup[i] -= 2

    

    if pd.notnull(poke1type2) and pd.notnull(poke2type2) and poke1type2 in weaknesses.ix[poke2type2, 'weaknesses']:

        type_matchup[i] -= 1 

    elif pd.notnull(poke1type2) and pd.notnull(poke2type2) and poke1type2 in weaknesses.ix[poke2type2, 'immunes']:

        type_matchup[i] -= 2

    

    #is poke 1 strong vs poke 2

    if poke2type1 in weaknesses.ix[poke1type1, 'strengths']:

        type_matchup[i] += 1 

    if pd.notnull(poke1type2) and poke2type1 in weaknesses.ix[poke1type2, 'strengths']:

        type_matchup[i] += 1

        

    if pd.notnull(poke2type2) and poke2type2 in weaknesses.ix[poke1type1, 'strengths']:

        type_matchup[i] += 1 

    if pd.notnull(poke1type2) and pd.notnull(poke2type2) and poke2type2 in weaknesses.ix[poke1type2, 'strengths']:

        type_matchup[i] += 1 

        

        #is poke 1 strong vs poke 2

    if poke1type1 in weaknesses.ix[poke2type1, 'strengths']:

        type_matchup[i] += 1 

    if pd.notnull(poke2type2) and poke1type1 in weaknesses.ix[poke2type2, 'strengths']:

        type_matchup[i] += 1        

    if pd.notnull(poke1type2) and poke1type2 in weaknesses.ix[poke2type1, 'strengths']:

        type_matchup[i] += 1 

    if pd.notnull(poke1type2) and pd.notnull(poke2type2) and poke1type2 in weaknesses.ix[poke2type2, 'strengths']:

        type_matchup[i] += 1
print(type_matchup)
print(first_poke.head())

print(second_poke.head())
#taking the numerical values of the dataset, we'll deal with the other values separately

first_poke_num = first_poke.iloc[:,2:8]

second_poke_num = second_poke.iloc[:,2:8]

first_poke_num.head()
#combine the two number rows through subtraction

poke_num = second_poke_num - first_poke_num

poke_num.head()
#all the columns put together we have:

poke_df = pd.concat([pd.Series(winners,name='winners'),pd.Series(type_matchup,name='type_matchup'), poke_num], axis=1)
poke_df.head(20)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
forest = RandomForestClassifier()

train_set, test_set = train_test_split(poke_df, test_size=0.4)

test_set, valid_set = train_test_split(test_set, test_size=0.5)
forest.fit(X=train_set.iloc[:,1:], y=train_set.iloc[:,0])
forest.score(X=test_set.iloc[:,1:], y=test_set.iloc[:,0])