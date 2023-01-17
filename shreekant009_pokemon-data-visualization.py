import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

pd.set_option('display.max_rows',20000, 'display.max_columns',100)

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')
pokemon = pd.read_csv("../input/complete-pokemon-dataset-updated-090420/pokedex_(Update.04.20).csv")

pokemon.info()
print(pokemon['generation'].nunique(),pokemon['generation'].unique())

print(pokemon['is_sub_legendary'].nunique(),pokemon['is_sub_legendary'].unique())

print(pokemon['is_legendary'].nunique(),pokemon['is_legendary'].unique())

print(pokemon['is_mythical'].nunique(),pokemon['is_mythical'].unique())

print(pokemon['species'].nunique(),pokemon['species'].unique())

print(pokemon['type_number'].nunique(),pokemon['type_number'].unique())

print(pokemon['type_1'].nunique(),pokemon['type_1'].unique())

print(pokemon['type_2'].nunique(),pokemon['type_2'].unique())

print(pokemon['height_m'].nunique(),pokemon['height_m'].unique())

print(pokemon['weight_kg'].nunique(),pokemon['weight_kg'].unique())

print(pokemon['abilities_number'].nunique(),pokemon['abilities_number'].unique())

print(pokemon['ability_1'].nunique(),pokemon['ability_1'].unique())

print(pokemon['ability_2'].nunique(),pokemon['ability_2'].unique())

print(pokemon['ability_hidden'].nunique(),pokemon['ability_hidden'].unique())
print(pokemon[pokemon['species'].isnull()].index.tolist())

print(pokemon[pokemon['type_1'].isnull()].index.tolist())

print(pokemon[pokemon['height_m'].isnull()].index.tolist())

print(pokemon[pokemon['weight_kg'].isnull()].index.tolist())
pokemon.at[240,'height_m']= 2.01

pokemon.at[240,'weight_kg']= 80.0

pokemon.at[240,['species','type_1','type_1']] = ['Royal Pokémon','Psychic','Water']

pokemon.at[1027,'weight_kg']= 950.0

pokemon = pokemon.drop(['Unnamed: 0','german_name','japanese_name'],axis=1)
#Top 50 Pokemon with their heights in meters

pokemon_height = pokemon.groupby('name')['height_m'].sum().reset_index().sort_values('height_m',ascending =False)

fig = px.bar(pokemon_height[:50], y='height_m', x='name', color='height_m', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their height')

fig.show()
#Top 50 Pokemon with their weights

pokemon_weight = pokemon.groupby('name')['weight_kg'].sum().reset_index().sort_values('weight_kg',ascending =False)

fig = px.bar(pokemon_weight[:50], y='weight_kg', x='name', color='weight_kg', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their weights')

fig.show()
#Diffrent Species of Pokemon with their Count

fig = px.treemap(pokemon, path=['species'],hover_data=['species'],color='species')

fig.update_layout(

    title='Statewise Urban Population and their Hospital Facilities for Covid19')

fig.update_layout(

    title='Diffrent Species of Pokemon with their Count')

fig.show()
#Diffrent Type of Pokemon with their Name

fig = px.treemap(pokemon, path=['type_1','name'],hover_data=['generation','type_1','is_sub_legendary','is_legendary','height_m','weight_kg'],color='type_1')

fig.update_layout(

    title='Diffrent Type of Pokemon with their Name')

fig.show()
#Sub Legendary Pokemon with their Total Points

pokemon_sub_legendary = pokemon.groupby('name')['is_sub_legendary','total_points'].sum().reset_index().sort_values('total_points',ascending =False)

pokemon_sub_legendary= pokemon_sub_legendary[pokemon_sub_legendary['is_sub_legendary']>0]

fig = px.bar(pokemon_sub_legendary, y='total_points', x='name', color='total_points', height=600)

fig.update_layout(

    title='Sub Legendary Pokemon with their Total Points')

fig.show()
#Legendary Pokemon with their Total Points

pokemon_legendary = pokemon.groupby('name')['is_legendary','total_points'].sum().reset_index().sort_values('total_points',ascending =False)

pokemon_legendary= pokemon_legendary[pokemon_legendary['is_legendary']>0]

fig = px.bar(pokemon_legendary, y='total_points', x='name', color='total_points', height=600)

fig.update_layout(

    title='Legendary Pokemon with their Total Points')

fig.show()
#Mythical Pokemon with their Total Points

pokemon_legendary = pokemon.groupby('name')['is_mythical','total_points'].sum().reset_index().sort_values('total_points',ascending =False)

pokemon_legendary= pokemon_legendary[pokemon_legendary['is_mythical']>0]

fig = px.bar(pokemon_legendary, y='total_points', x='name', color='total_points', height=600)

fig.update_layout(

    title='Mythical Pokemon with their Total Points')

fig.show()
#Pokemon with Different Type of Ability

fig = px.histogram(pokemon, x="abilities_number",color ='abilities_number')

fig.update_layout(

    title='Pokemon with Different Type of Ability')

fig.show()
#Ability of Pokemon

pokemon[['ability_2','ability_hidden']] = pokemon[['ability_2','ability_hidden']].fillna(value='NO INFO')

pokemon_ablity = pokemon.groupby(['name','abilities_number'])['ability_1','ability_2','ability_hidden'].sum().reset_index().sort_values('name',ascending =False)

pokemon_ablity

#Pokemon with their Abilities

fig = px.scatter_3d(pokemon_ablity, x="ability_1", y="ability_2", z="ability_hidden", color="abilities_number", size="abilities_number", hover_name="name",

                  symbol="abilities_number")

fig.update_layout(coloraxis_colorbar=dict(

    title="Ability of Pokemon",

    tickvals=[1,2,3],

    ticktext=["ability_1","ability_2","ability_hidden"],

    lenmode="pixels", len=150,

))

fig.update_layout(

    title='Pokemon with Different Type of Ability')

fig.show()
pokemon_total_points = pokemon.groupby('name')['total_points'].sum().reset_index().sort_values('total_points',ascending =False)

fig = px.bar(pokemon_total_points[:50], y='total_points', x='name', color='total_points', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their total_points')

fig.show()
fig = px.box(pokemon_total_points, y="total_points")

fig.update_layout(

    title='Range of Pokemon with their Total Points')

fig.show()
#Top 50 Pokemon with their HP

pokemon_HP = pokemon.groupby('name')['hp'].sum().reset_index().sort_values('hp',ascending =False)

fig = px.bar(pokemon_HP[:50], y='hp', x='name', color='hp', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their HP')

fig.show()
fig = px.box(pokemon_HP, y="hp")

fig.update_layout(

    title='Range of Pokemon with their HP')

fig.show()
pokemon_attack = pokemon.groupby('name')['attack'].sum().reset_index().sort_values('attack',ascending =False)

fig = px.bar(pokemon_attack[:50], y='attack', x='name', color='attack', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their Attack')

fig.show()
fig = px.box(pokemon_attack, y="attack")

fig.update_layout(

    title='Range of Pokemon with their Attack')

fig.show()
pokemon_defense = pokemon.groupby('name')['defense'].sum().reset_index().sort_values('defense',ascending =False)

fig = px.bar(pokemon_defense[:50], y='defense', x='name', color='defense', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their defense')

fig.show()
fig = px.box(pokemon_defense, y="defense")

fig.update_layout(

    title='Range of Pokemon with their Defense')

fig.show()
pokemon_sp_attack = pokemon.groupby('name')['sp_attack'].sum().reset_index().sort_values('sp_attack',ascending =False)

fig = px.bar(pokemon_sp_attack[:50], y='sp_attack', x='name', color='sp_attack', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their sp_attack')

fig.show()
fig = px.box(pokemon_sp_attack, y="sp_attack")

fig.update_layout(

    title='Range of Pokemon with their SP Attack')

fig.show()
pokemon_sp_defense = pokemon.groupby('name')['sp_defense'].sum().reset_index().sort_values('sp_defense',ascending =False)

fig = px.bar(pokemon_sp_defense[:50], y='sp_defense', x='name', color='sp_defense', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their sp_defense')

fig.show()
fig = px.box(pokemon_sp_defense, y="sp_defense")

fig.update_layout(

    title='Range of Pokemon with their SP Defense')

fig.show()
pokemon_speed = pokemon.groupby('name')['speed'].sum().reset_index().sort_values('speed',ascending =False)

fig = px.bar(pokemon_speed[:50], y='speed', x='name', color='speed', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their speed')

fig.show()
fig = px.box(pokemon_speed, y="speed")

fig.update_layout(

    title='Range of Pokemon with their Speed')

fig.show()
pokemon_catch_rate = pokemon.groupby('name')['catch_rate'].sum().reset_index().sort_values('catch_rate',ascending =False)

fig = px.bar(pokemon_catch_rate[:100], y='catch_rate', x='name', color='catch_rate', height=600)

fig.update_layout(

    title='Top 100 Pokemon with their Catch Rate')

fig.show()
fig = px.box(pokemon_catch_rate, y="catch_rate")

fig.update_layout(

    title='Range of Pokemon with their Catch Rate')

fig.show()
pokemon_base_friendship = pokemon.groupby('name')['base_friendship'].sum().reset_index().sort_values('base_friendship',ascending =False)

fig = px.bar(pokemon_base_friendship[:50], y='base_friendship', x='name', color='base_friendship', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their Base Friendship')

fig.show()
fig = px.box(pokemon_base_friendship, y="base_friendship")

fig.update_layout(

    title='Range of Pokemon with their base_friendship')

fig.show()
pokemon_base_experience = pokemon.groupby('name')['base_experience'].sum().reset_index().sort_values('base_experience',ascending =False)

fig = px.bar(pokemon_base_experience[:50], y='base_experience', x='name', color='base_experience', height=600)

fig.update_layout(

    title='Top 50 Pokemon with their Base Experience')

fig.show()
fig = px.box(pokemon_base_experience, y="base_experience")

fig.update_layout(

    title='Range of Pokemon with their Base Experience')

fig.show()
#Different Type of Growth Rate

fig = px.histogram(pokemon.dropna(), x="growth_rate",color ='growth_rate')

fig.update_layout(

    title='Different Type of Growth Rate')

fig.show()