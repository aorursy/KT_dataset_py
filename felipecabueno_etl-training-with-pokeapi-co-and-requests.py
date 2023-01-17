import pandas as pd



import requests

import json



import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('darkgrid')



color = 'deeppink'
url = 'https://pokeapi.co/api/v2/pokemon/'

response = requests.get(url)

content = response.json()

status_code = response.status_code



print('Connection status: {}'.format(status_code))

print('Total pokemons: {}'.format(content['count']))
page = 'https://pokeapi.co/api/v2/pokemon/?offset=0&limit=20'

content['next'] = 0



poke_names = []



while content['next'] != None:

    response = requests.get(page)

    content = response.json()



    for i in content['results']:

        poke_names.append(i)



    page = content['next']



names = pd.DataFrame(poke_names)
poke_content = []



for i in names['url']:

    response = requests.get(i)

    content_json = response.json()



    poke_content.append(content_json)



content = pd.DataFrame(poke_content)
pokemon = pd.concat([names, 

                     content.drop('name', axis= 1)], 

                    axis=1)



pokemon
cols_delete = ['id', 'base_experience', 'url', 'forms', 'game_indices',

               'held_items', 'is_default', 'location_area_encounters',

               'order', 'species', 'sprites']

pokemon.drop(cols_delete, axis= 1, inplace= True)

pokemon
abilities = []



for i in pokemon['abilities']:

    ability = []

    

    for j in i:

        ability.append(j['ability']['name'])

        

    abilities.append(ability)



pokemon['abilities'] = abilities
moves = []



for i in pokemon['moves']:

    move = []

    

    for j in i:

        move.append(j['move']['name'])

        

    moves.append(move)



pokemon['moves'] = moves
stats = []



for i in pokemon['stats']:

    base_stat = []

    

    for j in i:

        base_stat.append(j['base_stat'])

        

    stats.append(base_stat)



stats_df = pd.DataFrame(stats,

                        columns= ['hp',

                                  'attack',

                                  'defense',

                                  'special-attack',

                                  'special-defense',

                                  'speed'])



pokemon = pd.concat([pokemon, stats_df], 

                    axis=1)



pokemon.drop('stats', axis= 1, inplace= True)
types = []



for i in pokemon['types']:

    type_ = []

    

    for j in i:

        type_.append(j['type']['name'])

        

    types.append(type_)



types = pd.DataFrame(types, 

                     columns= ['type_1', 'type_2'])



pokemon = pd.concat([pokemon, types], 

                    axis=1)



pokemon.drop('types', axis= 1, inplace= True)
pokemon['base_total'] = (pokemon['hp']

                         + pokemon['attack']

                         + pokemon['defense']

                         + pokemon['special-attack']

                         + pokemon['special-defense']

                         + pokemon['speed'])
pokemon['height'] = pokemon['height']/10
pokemon['weight'] = pokemon['weight']/10
pokemon = pokemon[['name',

                   'height', 'weight',

                   'type_1', 'type_2',

                   'hp', 'attack',

                   'defense', 'special-attack',

                   'special-defense', 'speed',

                   'base_total', 'abilities',

                   'moves']]
pokemon
fig, ax = plt.subplots(figsize = (10, 12))



plt.subplot(2,1,1)

sns.scatterplot(x= 'height',

                y= 'weight',

                data= pokemon,

                color= color)



plt.subplots_adjust(hspace= 0.3)

plt.show()
fig, ax = plt.subplots(figsize = (10, 12))



plt.subplot(2,1,1)

sns.boxplot(x= 'type_1',

            y= 'base_total',

            data= pokemon,

            width= 0.3,

            color= color)

plt.xticks(rotation= 90)



plt.subplot(2,1,2)

sns.boxplot(x= 'type_2',

            y= 'base_total',

            data= pokemon,

            width= 0.3,

            color= color)

plt.xticks(rotation= 90)



plt.subplots_adjust(hspace= 0.3)

plt.show()
def dist_atr(dataframe, column):

    sns.kdeplot(dataframe[column],

                color= color)



fig, ax = plt.subplots(figsize = (10, 12))



plt.subplot(3,2,1)

dist_atr(pokemon, 'hp')



plt.subplot(3,2,2)

dist_atr(pokemon, 'attack')



plt.subplot(3,2,3)

dist_atr(pokemon, 'defense')



plt.subplot(3,2,4)

dist_atr(pokemon, 'special-attack')



plt.subplot(3,2,5)

dist_atr(pokemon, 'special-defense')



plt.subplot(3,2,6)

dist_atr(pokemon, 'speed')



plt.subplots_adjust(hspace= 0.3)

plt.show()
fig, ax = plt.subplots(figsize = (10, 6))



dist_atr(pokemon, 'base_total')



plt.subplots_adjust(hspace= 0.3)

plt.show()
pokemon.to_csv('pokemon.csv',

               index=False) 