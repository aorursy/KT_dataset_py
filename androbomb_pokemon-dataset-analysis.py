###############################################################

# NB: shift + tab HOLD FOR 2 SECONDS!

###############################################################





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print('\n ')

print('Getting traing dataset...')

data = pd.read_csv('../input/pokemon/Pokemon.csv')

print('Traing data set obtained. \n')
data.head(3)
plt.figure(figsize=(16,8))

b = sns.countplot(x = 'Type 1', data=data, order = data['Type 1'].value_counts().index)

b.axes.set_title("Distribution of Types as Type 1",fontsize=30)

b.set_xlabel("Type",fontsize=18)

b.set_ylabel("# of Pokemons",fontsize=18)



plt.figure(figsize=(16,8))

c = sns.countplot(x = 'Type 2', data=data, order = data['Type 1'].value_counts().index)

c.axes.set_title("Distribution of Types as Type 2",fontsize=30)

c.set_xlabel("Type",fontsize=18)

c.set_ylabel("# of Pokemons",fontsize=18)

plt.show()
types = ['Normal', 'Fire', 'Fighting', 'Water', 'Flying', 'Grass', 'Poison', 'Electric', 'Ground', 'Psychic', 'Rock', 'Ice', 'Bug', 'Dragon', 'Ghost', 'Dark', 'Steel', 'Fairy']



x1 = data['Type 1'].value_counts()

x2 = data['Type 2'].value_counts()

A = np.zeros((len(types), len(types)))

df=data.fillna('NaN')

for i in range(0, len(types)):

    for j in range(0, len(types)):

        if i!=j : # by definition, it doesn't exist a pkmn that have type 1 = type 2 

            if ((data[(data['Type 1']==types[i]) & (data['Type 2']==types[j])]['Type 1'].value_counts()).empty) :

                A[i,j] = 0

            else :

                A[i,j] = data[(data['Type 1']==types[i]) & (data['Type 2']==types[j])]['Type 1'].value_counts()[0]

        else : # we replace the diagonal column with the total count of type 1 pkmns having NO type 2

            A[i,j] =  data[(data['Type 1']==types[i]) & (df['Type 2'] == 'NaN')]['Type 1'].value_counts()[0]





plt.figure(figsize=(13,10))

b = sns.heatmap(A, annot=True, linewidths=.5, cmap="YlGnBu", xticklabels=types, yticklabels=types)

b.axes.set_title("Distribution of Types",fontsize=30)

b.set_xlabel("Type 1",fontsize=18)

b.set_ylabel("Type 2",fontsize=18)

plt.show()
data.sort_values('Total', ascending = False).head(10)
print('The top 5 pokemon by HP \n')

print(data.sort_values('HP', ascending = False)[['Name', 'HP']].head(10))

print('\n')



print('The top 5 pokemon by Attack \n')

print(data.sort_values('Attack', ascending = False)[['Name', 'Attack']].head(10))

print('\n')



print('The top 5 pokemon by Defense \n')

print(data.sort_values('Defense', ascending = False)[['Name', 'Defense']].head(10))

print('\n')



print('The top 5 pokemon by Sp. Attack \n')

print(data.sort_values('Sp. Atk', ascending = False)[['Name', 'Sp. Atk']].head(10))

print('\n') 



print('The top 5 pokemon by Sp. Defense \n')

print(data.sort_values('Sp. Def', ascending = False)[['Name', 'Sp. Def']].head(10))

print('\n')



print('The top 5 pokemon by Speed \n')

print(data.sort_values('Speed', ascending = False)[['Name', 'Speed']].head(10))

print('\n')
data[data['Legendary']==False].sort_values('Total', ascending = False).head(10)
print('The top 5 pokemon by HP \n')

print(data[data['Legendary']==False].sort_values('HP', ascending = False)[['Name', 'HP']].head(10))

print('\n')



print('The top 5 pokemon by Attack \n')

print(data[data['Legendary']==False].sort_values('Attack', ascending = False)[['Name', 'Attack']].head(10))

print('\n')



print('The top 5 pokemon by Defense \n')

print(data[data['Legendary']==False].sort_values('Defense', ascending = False)[['Name', 'Defense']].head(10))

print('\n')



print('The top 5 pokemon by Sp. Attack \n')

print(data[data['Legendary']==False].sort_values('Sp. Atk', ascending = False)[['Name', 'Sp. Atk']].head(10))

print('\n') 



print('The top 5 pokemon by Sp. Defense \n')

print(data[data['Legendary']==False].sort_values('Sp. Def', ascending = False)[['Name', 'Sp. Def']].head(10))

print('\n')



print('The top 5 pokemon by Speed \n')

print(data[data['Legendary']==False].sort_values('Speed', ascending = False)[['Name', 'Speed']].head(10))

print('\n')
df = data.drop('Name', axis=1).drop('#', axis=1).drop('Type 1', axis=1).drop('Type 2', axis=1).drop('Generation', axis=1)



plt.figure(figsize=(16,8))

ax = sns.pairplot(df, hue='Legendary', markers=['o', 's'], kind="reg", vars=['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'])

plt.show()
import plotly.express as px



df = 1*data.drop('Name', axis=1).drop('#', axis=1).drop('Type 1', axis=1).drop('Type 2', axis=1).drop('Generation', axis=1)

    # multiply the dataframe by 1 transforms the True/False in the 'Legendary' column into 1/0.

    

fig = px.scatter_matrix(df,

    dimensions=['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'],

                        symbol = "Legendary", color='Legendary',

                        title = 'Correlations in PKMN Stats')

fig.update_traces(diagonal_visible=False)

fig.show()
plt.figure(figsize=(15,8))

b = sns.boxplot(x='Total', y='Type 1', data=data)

b.axes.set_title("Total by Type 1",fontsize=30)

b.set_xlabel("Total",fontsize=18)

b.set_ylabel("Type 1",fontsize=18)

plt.show()
plt.figure(figsize=(15,8))

b = sns.boxplot(x='Total', y='Type 1', data=data, hue='Legendary')

b.axes.set_title("Total by Type 1",fontsize=30)

b.set_xlabel("Total",fontsize=18)

b.set_ylabel("Type 1",fontsize=18)

plt.show()
# HP

plt.figure(figsize=(15,8))

c = sns.boxplot(x='HP', y='Type 1', data=data)

c.axes.set_title("HP by Type 1",fontsize=30)

c.set_xlabel("HP",fontsize=18)

c.set_ylabel("Type 1",fontsize=18)



# Attack

plt.figure(figsize=(15,8))

c = sns.boxplot(x='Attack', y='Type 1', data=data)

c.axes.set_title("Attack by Type 1",fontsize=30)

c.set_xlabel("Attack",fontsize=18)

c.set_ylabel("Type 1",fontsize=18)



# Defense

plt.figure(figsize=(15,8))

c = sns.boxplot(x='Defense', y='Type 1', data=data)

c.axes.set_title("Defense by Type 1",fontsize=30)

c.set_xlabel("Total",fontsize=18)

c.set_ylabel("Type 1",fontsize=18)





# Sp. Atk

plt.figure(figsize=(15,8))

c = sns.boxplot(x='Sp. Atk', y='Type 1', data=data)

c.axes.set_title("Sp. Atk by Type 1",fontsize=30)

c.set_xlabel("Sp. Atk",fontsize=18)

c.set_ylabel("Type 1",fontsize=18)





# Sp. Def

plt.figure(figsize=(15,8))

c = sns.boxplot(x='Sp. Def', y='Type 1', data=data)

c.axes.set_title("Sp. Def by Type 1",fontsize=30)

c.set_xlabel("Sp. Def",fontsize=18)

c.set_ylabel("Type 1",fontsize=18)



# Speed

plt.figure(figsize=(15,8))

c = sns.boxplot(x='Speed', y='Type 1', data=data)

c.axes.set_title("Speed by Type 1",fontsize=30)

c.set_xlabel("Speed",fontsize=18)

c.set_ylabel("Type 1",fontsize=18)



plt.show()
types = ['Normal', 'Fire', 'Fighting', 'Water', 'Flying', 'Grass', 'Poison', 'Electric', 'Ground', 'Psychic', 'Rock', 'Ice', 'Bug', 'Dragon', 'Ghost', 'Dark', 'Steel', 'Fairy']

stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']



for tp in types:

    plt.figure(figsize=(16,8))

    for st in stats:

        c = sns.distplot(data[data['Type 1'] == tp][st], label=st, kde=False)

        c.set_xlabel(tp, fontsize=20)

        c.set_ylabel('Stats',fontsize=20)

    plt.legend(fontsize=15)

    plt.show()
types = ['Normal', 'Fire', 'Fighting', 'Water', 'Flying', 'Grass', 'Poison', 'Electric', 'Ground', 'Psychic', 'Rock', 'Ice', 'Bug', 'Dragon', 'Ghost', 'Dark', 'Steel', 'Fairy']

stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']



for st in stats:

    plt.figure(figsize=(16,8))

    for tp in types:

        c = sns.distplot(data[data['Type 1'] == tp][st], label=tp, kde=False)

        c.set_xlabel(st, fontsize=20)

        c.set_ylabel('# of Pokemon in bin',fontsize=20)

    # Put a legend to the right of the current axis

    plt.legend(fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
types = ['Normal', 'Fire', 'Fighting', 'Water', 'Flying', 'Grass', 'Poison', 'Electric', 'Ground', 'Psychic', 'Rock', 'Ice', 'Bug', 'Dragon', 'Ghost', 'Dark', 'Steel', 'Fairy']

stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']



for st in stats:

    plt.figure(figsize=(16,8))

    for tp in types:

        c = sns.distplot(data[data['Type 1'] == tp][st], label=tp, kde=True, hist=False)

        c.set_xlabel(st, fontsize=20)

        c.set_ylabel('% of Pokemon',fontsize=20)

    # Put a legend to the right of the current axis

    plt.legend(fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()
plt.figure(figsize=(16,8))

b = sns.countplot(x = 'Type 1', data=data[data['Legendary']==True], order = data['Type 1'].value_counts().index)

b.axes.set_title("Distribution of Types as Type 1",fontsize=30)

b.set_xlabel("Type",fontsize=18)

b.set_ylabel("# of Pokemons",fontsize=18)



plt.figure(figsize=(16,8))

c = sns.countplot(x = 'Type 2', data=data[data['Legendary']==True], order = data['Type 1'].value_counts().index)

c.axes.set_title("Distribution of Types as Type 2",fontsize=30)

c.set_xlabel("Type",fontsize=18)

c.set_ylabel("# of Pokemons",fontsize=18)



plt.show()
plt.figure(figsize=(18, 6))

b = sns.boxplot(x='Type 1', y='Total', data=data, hue='Legendary')

b.axes.set_title("Distribution of Total by Type 1",fontsize=30)

b.set_xlabel("Type 1",fontsize=18)

b.set_ylabel("Total",fontsize=18)

plt.show()
plt.figure(figsize=(18, 6))

b = sns.violinplot(x='Type 1', y='Total', data=data, hue='Legendary', split=True)

b.axes.set_title("Distribution of Total by Type 1",fontsize=30)

b.set_xlabel("Type 1",fontsize=20)

b.set_ylabel("Total",fontsize=20)

plt.show()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

from plotly import tools

import plotly.graph_objs as go
# Function that plots the stat of a pkmn

def hex_stat(data, xx):

    x = data[data['Name'] == xx]

    data = [go.Scatterpolar(

        r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],

        theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],fill = 'toself')]



    layout = go.Layout(polar = dict(

        radialaxis = dict(

            visible = True,

            range = [0, 250]

        )

    ),showlegend = False,

                       title = "Stats of {}".format(x.Name.values[0]))

    

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename = "Single Pokemon stats")
data.sort_values('Total', ascending = False).head(1)
hex_stat(data, 'RayquazaMega Rayquaza')
# Creating a method to compare 2 pokemon

def compare2pokemon(data, xx,yy):

    x = data[data['Name'] == xx]

    y = data[data['Name'] == yy]



    trace0 = go.Scatterpolar(

      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],

      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],

      fill = 'toself',

      name = x.Name.values[0]

    )



    trace1 = go.Scatterpolar(

      r = [y['HP'].values[0],y['Attack'].values[0],y['Defense'].values[0],y['Sp. Atk'].values[0],y['Sp. Def'].values[0],y['Speed'].values[0],y["HP"].values[0]],

      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],

      fill = 'toself',

      name = y.Name.values[0]

    )



    data = [trace0, trace1]



    layout = go.Layout(

      polar = dict(

        radialaxis = dict(

          visible = True,

          range = [0, 200]

        )

      ),

      showlegend = True,

      title = "{} vs {}".format(x.Name.values[0],y.Name.values[0])

    )

    fig = go.Figure(data=data, layout=layout)

    iplot(fig, filename = "Two Pokemon stats")
compare2pokemon(data, "Mew","Mewtwo")
compare2pokemon(data, "Dragonite","Haxorus")
data[data['Legendary']==False].sort_values('Total', ascending = False).head(2)
compare2pokemon(data, "MetagrossMega Metagross","GarchompMega Garchomp")
data[data['Legendary']==False].sort_values('Total', ascending = True).head(2)
compare2pokemon(data, "Sunkern","Azurill")