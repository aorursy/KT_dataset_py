## Código gerado automaticamente pelo Kaggle.

## Incluídas importações de outras bibliotecas também.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
Pokedex = pd.read_csv('../input/Pokemon.csv')
Pokedex.head()
Pokedex.rename(columns={'Sp. Atk': 'SpAtk', 'Sp. Def': 'SpDef'}, inplace=True)
Pokedex.tail()
Pokedex.sample(5)
Pokedex.describe()
Pokedex.info()
Pokedex['Generation'].value_counts()
sns.catplot(x='Generation',kind='count',data=Pokedex)
labels = ['1', '2', '3', '4', '5', '6']

sizes = [166, 165, 160, 121, 106, 82]

plt.pie(x=sizes, labels=labels, autopct='%1.1f%%')

plt.title("Percentual de Pokémons de cada Geração")

fig=plt.gcf()

fig.set_size_inches(9,9)

plt.show()
maiorHP = Pokedex[Pokedex.HP  == Pokedex.HP.max()]

print('Pokémon com maior HP: ')

maiorHP
menorHP = Pokedex[Pokedex.HP  == Pokedex.HP.min()]

print('Pokémon com menor HP: ')

menorHP
maiorAttack = Pokedex[Pokedex.Attack  == Pokedex.Attack.max()]

print('Pokémon com maior Ataque: ')

maiorAttack
menorAttack = Pokedex[Pokedex.Attack  == Pokedex.Attack.min()]

print('Pokémon com menor Ataque: ')

menorAttack
maiorDefense = Pokedex[Pokedex.Defense  == Pokedex.Defense.max()]

print('Pokémon com maior Defesa: ')

maiorDefense
menorDefense = Pokedex[Pokedex.Defense  == Pokedex.Defense.min()]

print('Pokémon com menor Defesa: ')

menorDefense
maiorSpAtk = Pokedex[Pokedex['SpAtk']  == Pokedex['SpAtk'].max()]

print('Pokémon com maior Ataque Especial: ')

maiorSpAtk
menorSpAtk = Pokedex[Pokedex['SpAtk']  == Pokedex['SpAtk'].min()]

print('Pokémon com menor Ataque Especial: ')

menorSpAtk
maiorSpDef = Pokedex[Pokedex['SpDef']  == Pokedex['SpDef'].max()]

print('Pokémon com maior Defesa Especial: ')

maiorSpDef
menorSpDef = Pokedex[Pokedex['SpDef']  == Pokedex['SpDef'].min()]

print('Pokémon com menor Defesa Especial: ')

menorSpDef
maiorSpeed = Pokedex[Pokedex.Speed  == Pokedex.Speed.max()]

print('Pokémon com maior Velocidade: ')

maiorSpeed
menorSpeed = Pokedex[Pokedex.Speed  == Pokedex.Speed.min()]

print('Pokémon com menor Velocidade: ')

menorSpeed
Pokedex['Type 1'].value_counts()
sns.catplot(x='Type 1',kind='count',data=Pokedex, height=11)
labels = ['Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Rock', 'Electric', 'Ground', 'Dragon', 'Ghost', 'Dark'

, 'Poison', 'Fighting', 'Steel', 'Ice', 'Fairy', 'Flying']

sizes = [112, 98, 70, 69, 57, 52, 44, 44, 32, 32, 32, 31, 28, 27, 27, 24, 17, 4]

plt.pie(x=sizes, labels=labels, autopct='%1.1f%%')

plt.title('Percentual do tipo primário dos Pokémons')

fig=plt.gcf()

fig.set_size_inches(9,9)

plt.show()
Pokedex['Type 2'].value_counts()
sns.catplot(x='Type 2',kind='count',data=Pokedex, height=11)
labels = ['Flying', 'Ground', 'Poison', 'Psychic', 'Fighting', 'Grass', 'Fairy', 'Steel', 'Dark', 'Dragon', 'Ice', 'Ghost'

, 'Water', 'Rock', 'Fire', 'Electric', 'Normal', 'Bug'

]

sizes = [97, 35, 34, 33, 26, 25, 23, 22, 20, 18, 14, 14, 14, 14, 12, 6, 4, 3]

plt.pie(x=sizes, labels=labels, autopct='%1.1f%%')

plt.title('Percentual do tipo secundário dos Pokémons')

fig=plt.gcf()

fig.set_size_inches(9,9)

plt.show()
lendarios = Pokedex[Pokedex.Legendary  == 1]

print('Pokémons lendários: ')

lendarios
maiorTotalLegen = lendarios[lendarios.Total  == lendarios.Total.max()]

print('Pokémon lendário com maior Total de Status: ')

maiorTotalLegen
menorTotalLegen = lendarios[lendarios.Total  == lendarios.Total.min()]

print('Pokémon lendário com menor Total de Status: ')

menorTotalLegen
Pokedex['Legendary'].value_counts()
sns.catplot(x='Legendary',kind='count',data=Pokedex, height=7)
labels = ['Regular', 'Legendary']

sizes = [735, 65]

plt.pie(x=sizes, labels=labels, autopct='%1.1f%%')

plt.title('Percentual do tipo secundário dos Pokémons')

fig=plt.gcf()

fig.set_size_inches(9,9)

plt.show()
## Biblioteca utilizada para ignorar warnings de execuções.

import warnings

# from beakerx import *

warnings.filterwarnings('ignore')
plt.figure(figsize=(10, 5))

HP = sns.distplot(Pokedex.HP, kde=False)

HP.set_title('HP')
plt.figure(figsize=(10, 5))

Attack = sns.distplot(Pokedex.Attack, kde=False)

Attack.set_title('Attack')
plt.figure(figsize=(10, 5))

Defense = sns.distplot(Pokedex.Defense, kde=False)

Defense.set_title('Defense')
plt.figure(figsize=(10, 5))

SpAtk = sns.distplot(Pokedex.SpAtk, kde=False)

SpAtk.set_title('SpAtk')
plt.figure(figsize=(10, 5))

SpDef = sns.distplot(Pokedex.SpDef, kde=False)

SpDef.set_title('SpDef')
plt.figure(figsize=(10, 5))

Speed = sns.distplot(Pokedex.Speed, kde=False)

Speed.set_title('Speed')