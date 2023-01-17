import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot

import matplotlib.pyplot as plt

from io import StringIO
%matplotlib inline
pokemon = pd.read_csv('../input/Pokemon.csv')

pokemon = pokemon[pokemon['#'] < 151]
pokemon_attack=pokemon[['Name', 'Type 1', 'Type 2']]

pokemon_attack['Normal']=0

pokemon_attack['Fire']=0

pokemon_attack['Water']=0

pokemon_attack['Electric']=0

pokemon_attack['Ice']=0

pokemon_attack['Fighting']=0

pokemon_attack['Ground']=0

pokemon_attack['Poison']=0

pokemon_attack['Ghost']=0

pokemon_attack['Flying']=0

pokemon_attack['Psychic']=0

pokemon_attack['Fairy']=0

pokemon_attack['Steel']=0

pokemon_attack['Bug']=0

pokemon_attack['Dragon']=0

pokemon_attack['Rock']=0

pokemon_attack['Dark']=0

pokemon_attack['Grass']=0

attack_power = pd.read_csv(StringIO("""Attack,Normal,Fire,Water,Electric,Grass,Ice,Fighting,Poison,Ground,Flying,Psychic,Bug,Rock,Ghost,Dragon,Dark,Steel,Fairy

Normal,1,1,1,1,1,1,1,1,1,1,1,1,0.5,0,1,1,0.5,1

Fire,1,0.5,0.5,1,2,2,1,1,1,1,1,2,0.5,1,0.5,1,2,1

Water,1,2,0.5,1,0.5,1,1,1,2,1,1,1,2,1,0.5,1,1,1

Electric,1,1,2,0.5,0.5,1,1,1,0,2,1,1,1,1,0.5,1,1,1

Grass,1,0.5,2,1,0.5,1,1,0.5,2,0.5,1,0.5,2,1,0.5,1,0.5,1

Ice,1,0.5,0.5,1,2,0.5,1,1,2,2,1,1,1,1,2,1,0.5,1

Fighting,2,1,1,1,1,2,1,0.5,1,0.5,0.5,0.5,2,0,1,2,2,0.5

Poison,1,1,1,1,2,1,1,0.5,0.5,1,1,1,0.5,0.5,1,1,0,2

Ground,1,2,1,2,0.5,1,1,2,1,0,1,0.5,2,1,1,1,2,1

Flying,1,1,1,0.5,2,1,2,1,1,1,1,2,0.5,1,1,1,0.5,1

Psychic,1,1,1,1,1,1,2,2,1,1,0.5,1,1,1,1,0,0.5,1

Bug,1,0.5,1,1,2,1,0.5,0.5,1,0.5,2,1,1,0.5,1,2,0.5,0.5

Rock,1,2,1,1,1,2,0.5,1,0.5,2,1,2,1,1,1,1,0.5,1

Ghost,0,1,1,1,1,1,1,1,1,1,2,1,1,2,1,0.5,1,1

Dragon,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,0.5,0

Dark,1,1,1,1,1,1,0.5,1,1,1,2,1,1,2,1,0.5,1,0.5

Steel,1,0.5,0.5,0.5,1,2,1,1,1,1,1,1,2,1,1,1,0.5,2

Fairy,1,0.5,1,1,1,1,2,0.5,1,1,1,1,1,1,2,2,0.5,1"""))
attack_power.set_index('Attack', inplace=True)
type_list=['Normal','Fire','Water','Electric','Grass','Ice','Fighting','Poison','Ground','Flying','Psychic','Bug','Rock','Ghost','Dragon','Dark','Steel','Fairy']
m=0

for i in pokemon['Name']:

    type_of_pokemon=pokemon['Type 1'].iloc[m]

    for j in type_list:

        pokemon_attack[j].iloc[m]=((pokemon['Attack'].iloc[m]*attack_power[j][type_of_pokemon]) +

        (pokemon['Defense'].iloc[m]*attack_power[j][type_of_pokemon]) +

        (pokemon['Sp. Atk'].iloc[m]*attack_power[j][type_of_pokemon]) +

        (pokemon['Sp. Def'].iloc[m]*attack_power[j][type_of_pokemon]) +

        (pokemon['Speed'].iloc[m]*attack_power[j][type_of_pokemon]))

    m=m+1
pokemon_attack['Total_Strength']=(pokemon_attack['Normal']+pokemon_attack['Grass']+pokemon_attack['Fire']+

                                pokemon_attack['Water']+pokemon_attack['Bug']+pokemon_attack['Poison']+

                                pokemon_attack['Electric']+pokemon_attack['Ground']+pokemon_attack['Fairy']+

                                pokemon_attack['Fighting']+pokemon_attack['Psychic']+pokemon_attack['Rock']+

                                pokemon_attack['Ghost']+pokemon_attack['Ice']+pokemon_attack['Dragon']+

                                pokemon_attack['Dark']+pokemon_attack['Steel']+pokemon_attack['Flying'])
defender='Charmander'

select_pokemon=pd.DataFrame()

select_pokemon=pokemon_attack
type_of_defender = pokemon[pokemon['Name'] == defender]['Type 1'].iloc[0]
select_pokemon.sort_values(type_of_defender, ascending=False, inplace=True)

x=select_pokemon[[type_of_defender, 'Name', 'Type 1']].head(10)
dict={ 'Grass': '#8ED752','Fire': '#F95643' ,'Water': '#53AFFE','Bug': '#C3D221','Normal': '#BBBDAF',

      'Poison': '#AD5CA2' , 'Electric' :'#F8E64E','Ground': '#F0CA42' ,'Fairy': '#F9AEFE' ,

      'Fighting': '#A35449' ,'Psychic': '#FB61B4' , 'Rock': '#CDBD72' ,'Ghost': '#7673DA' ,

      'Ice': '#66EBFF','Dragon': '#8B76FF' ,'Dark': '#8E6856' , 'Steel': '#C3C1D7' ,'Flying': '#75A4F9' }
s1 = [0,0,0,0,0,0,0,0,0,0]

m=0

for i in x['Type 1']:

    s1[m]=dict[i]

    m=m+1

sns.set_palette(s1)

a4_dims = (12, 8)

fig, ax = pyplot.subplots(figsize=a4_dims)

fig.suptitle('Best 10 pokemons to attack with', fontsize=18, fontweight='bold')

sns.barplot(x=type_of_defender, y='Name', data=x, estimator=np.sum, palette=s1)

a4_dims = (8, 10)

fig, ax = pyplot.subplots(figsize=a4_dims)

fig.suptitle('Count of Types', fontsize=18, fontweight='bold')

sns.countplot(y='Type 1', data=pokemon_attack)      
a4_dims = (8, 10)

fig, ax = pyplot.subplots(figsize=a4_dims)

fig.suptitle('Distribution of pokemons according to strengths', fontsize=18, fontweight='bold')

sns.stripplot(y='Type 1',x='Total_Strength', data=pokemon_attack, jitter=False, hue='Type 1')
select_pokemon.sort_values('Total_Strength', ascending=False, inplace=True)

s1 = [0,0,0,0,0,0,0,0,0,0]

m=0

for i in select_pokemon['Type 1'].head(10):

    s1[m]=dict[i]

    m=m+1

sns.set_palette(s1)

a4_dims = (12, 8)

fig, ax = pyplot.subplots(figsize=a4_dims)

fig.suptitle('Top 20 Pokemons', fontsize=18, fontweight='bold')

sns.barplot(x='Total_Strength', y='Name', data=select_pokemon.head(20), estimator=np.sum, palette=s1)

select_pokemon.sort_values('Total_Strength', ascending=True, inplace=True)

s1 = [0,0,0,0,0,0,0,0,0,0]

m=0

for i in select_pokemon['Type 1'].head(10):

    s1[m]=dict[i]

    m=m+1

sns.set_palette(s1)

a4_dims = (12, 8)

fig, ax = pyplot.subplots(figsize=a4_dims)

fig.suptitle('Least Powerful: 20 Pokemons', fontsize=18, fontweight='bold')

sns.barplot(x='Total_Strength', y='Name', data=select_pokemon.head(20), estimator=np.sum, palette=s1)
Legendary_Pokemon=pd.DataFrame({"Name": range(pokemon[pokemon['Legendary']]['Name'].count())})

Legendary_Pokemon['Type']=0

Legendary_Pokemon['Strength']=0

m=0

c=0

for i in pokemon['Legendary']:

    if i==True:

        Legendary_Pokemon['Name'].iloc[c]=(pokemon['Name'].iloc[m])

        Legendary_Pokemon['Type'].iloc[c]=(pokemon['Type 1'].iloc[m])

        m=m+1

        c=c+1

    else:

        m=m+1



m=0; c=0;

for i in pokemon_attack['Name']:

    for j in Legendary_Pokemon['Name']:

        if(i==j):

            Legendary_Pokemon['Strength'].iloc[c]=pokemon_attack['Total_Strength'].iloc[m]

            c=c+1

            break

    m=m+1

    

l=[None]*c

m=0

for i in Legendary_Pokemon['Type']:

    l[m]=dict[i]

    m=m+1

sns.set_palette(l)

a4_dims = (12, 6)

fig, ax = pyplot.subplots(figsize=a4_dims)

fig.suptitle('Legendary Pokemons', fontsize=18, fontweight='bold')

sns.barplot(x='Strength', y='Name', data=Legendary_Pokemon, estimator=np.sum, palette=l)        

        

# pokemons stronger than atleast two Legendary Pokemons (Articuno and Zapdos)



m=0; c=0;

for i in pokemon_attack['Name']:

    if any(i in s for s in Legendary_Pokemon['Name']):

        m=m+1

    else:

        if ((pokemon_attack['Total_Strength'].iloc[m]) > (Legendary_Pokemon[Legendary_Pokemon['Name']=='Zapdos']['Strength'].iloc[0])):

            c=c+1

        m=m+1



Strong_Pokemon=pd.DataFrame({"Name": range(c)})

Strong_Pokemon['Type']=0

Strong_Pokemon['Strength']=0     



m=0; c=0;

for i in pokemon_attack['Name']:

    if any(i in s for s in Legendary_Pokemon['Name']):

        m=m+1

    else:

        if((pokemon_attack['Total_Strength'].iloc[m]) > (Legendary_Pokemon[Legendary_Pokemon['Name']=='Zapdos']['Strength'].iloc[0])):



            Strong_Pokemon['Name'].iloc[c]=pokemon_attack['Name'].iloc[m]

            Strong_Pokemon['Type'].iloc[c]=pokemon_attack['Type 1'].iloc[m]

            Strong_Pokemon['Strength'].iloc[c]=pokemon_attack['Total_Strength'].iloc[m]

            c=c+1   

        m=m+1



l=[None]*c

m=0

for i in Strong_Pokemon['Type']:

    l[m]=dict[i]

    m=m+1

sns.set_palette(l)

a4_dims = (12, 6)

fig, ax = pyplot.subplots(figsize=a4_dims)

fig.suptitle('Stronger than a few Legendary Pokemons', fontsize=18, fontweight='bold')

sns.barplot(x='Strength', y='Name', data=Strong_Pokemon, estimator=np.sum, palette=l)