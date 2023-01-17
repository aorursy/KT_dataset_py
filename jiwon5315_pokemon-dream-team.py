import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(10) 
pokemon['Generation'].describe()
first_gen = pokemon.loc[pokemon['Generation']==1]
first_gen.describe() 
# Find the pokemon with the biggest Total Points  
first_gen.loc[first_gen.Total == first_gen.Total.max()] 

# Both Mewtwo X and Y have 780, so we will use both data.
# We will find Top Five and include them in the list for comparison.
result = first_gen.sort_values(by=['Total'],ascending=False).head(5) 
# Find pokemons with the highest HP, Attack, Defense, Sp.Atk, Sp.Def, and Speed
result = result.append(first_gen.loc[first_gen.HP == first_gen.HP.max()])
result = result.append(first_gen.loc[first_gen.Attack == first_gen.Attack.max()])
result = result.append(first_gen.loc[first_gen.Defense == first_gen.Defense.max()])
result = result.append(first_gen.loc[first_gen['Sp. Atk'] == first_gen['Sp. Atk'].max()])
result = result.append(first_gen.loc[first_gen['Sp. Def'] == first_gen['Sp. Def'].max()])
result = result.append(first_gen.loc[first_gen.Speed == first_gen.Speed.max()])
result.reset_index().drop(['index'], axis=1) 
# Compare the pokemons based on all their specifications
# We will delete any duplicates
result = result.drop_duplicates(subset="Name") 
result
fig, axes = plt.subplots(figsize=(15,9), ncols=3, nrows=2)
sns.scatterplot(x="HP", y="Total", data=result, hue="Legendary", ax=axes[0][0])
sns.scatterplot(x="Attack", y="Total", data=result, hue="Legendary", ax=axes[0][1])
sns.scatterplot(x="Defense", y="Total", data=result, hue="Legendary", ax=axes[0][2])
sns.scatterplot(x="Sp. Atk", y="Total", data=result, hue="Legendary", ax=axes[1][0])
sns.scatterplot(x="Sp. Def", y="Total", data=result, hue="Legendary", ax=axes[1][1])
sns.scatterplot(x="Speed", y="Total", data=result, hue="Legendary", ax=axes[1][2])


plt.subplots_adjust(
    wspace  =  0.3, 
    hspace  =  0.3
)

# The highest Total (Mewtwo group) is the only legendary pokemon in the list, so it is highlighted for comparison.
legend = first_gen.loc[first_gen.Legendary == True] 
# Out of 166 First Generation pokemons, 6 are Legendary.
legend
# Let's compare the Legendary pokemons from the average pokemons
fig, axes = plt.subplots(figsize=(15,10), ncols=3, nrows=2)
sns.boxplot(x='Legendary', y='HP', data=first_gen, palette='magma', ax=axes[0][0])
sns.boxplot(x='Legendary', y='Attack', data=first_gen, palette='magma', ax=axes[0][1])
sns.boxplot(x='Legendary', y='Defense', data=first_gen, palette='magma', ax=axes[0][2])
sns.boxplot(x='Legendary', y='Sp. Atk', data=first_gen, palette='magma', ax=axes[1][0])
sns.boxplot(x='Legendary', y='Sp. Def', data=first_gen, palette='magma', ax=axes[1][1])
sns.boxplot(x='Legendary', y='Speed', data=first_gen, palette='magma', ax=axes[1][2])
fig, axes = plt.subplots(figsize=(15,18), ncols=2, nrows=3)
sns.violinplot(x=first_gen["Legendary"], y=first_gen["HP"], ax=axes[0][0], palette='BrBG')
sns.violinplot(x=first_gen["Legendary"], y=first_gen["Attack"], ax=axes[0][1], palette='BrBG')
sns.violinplot(x=first_gen["Legendary"], y=first_gen["Defense"], ax=axes[1][0], palette='BrBG')
sns.violinplot(x=first_gen["Legendary"], y=first_gen["Sp. Atk"], ax=axes[1][1], palette='BrBG')
sns.violinplot(x=first_gen["Legendary"], y=first_gen["Sp. Def"], ax=axes[2][0], palette='BrBG')
sns.violinplot(x=first_gen["Legendary"], y=first_gen["Speed"], ax=axes[2][1], palette='BrBG')
fig, axes = plt.subplots(figsize=(15,9), ncols=3, nrows=2)
sns.swarmplot(y="HP", data=first_gen, ax=axes[0][0], dodge=True)
sns.swarmplot(y="HP", data=legend, ax=axes[0][0], color='orange', dodge=True)
sns.swarmplot(y="Attack", data=first_gen, ax=axes[0][1], dodge=True)
sns.swarmplot(y="Attack", data=legend, ax=axes[0][1], color='orange', dodge=True)
sns.swarmplot(y="Defense", data=first_gen, ax=axes[0][2], dodge=True)
sns.swarmplot(y="Defense", data=legend, ax=axes[0][2], color='orange', dodge=True)
sns.swarmplot(y="Sp. Atk", data=first_gen, ax=axes[1][0], dodge=True)
sns.swarmplot(y="Sp. Atk", data=legend, ax=axes[1][0], color='orange', dodge=True)
sns.swarmplot(y="Sp. Def", data=first_gen, ax=axes[1][1], dodge=True)
sns.swarmplot(y="Sp. Def", data=legend, ax=axes[1][1], color='orange', dodge=True)
sns.swarmplot(y="Speed", data=first_gen, ax=axes[1][2], dodge=True)
sns.swarmplot(y="Speed", data=legend, ax=axes[1][2], color='orange', dodge=True)
 
plt.subplots_adjust(
    wspace  =  0.3, 
    hspace  =  0.3
) 
from scipy import stats

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(u"\u03C1 = {:.2f}".format(r), #unicode code for lowercase rho (ρ)
                xy=(.1, .9), xycoords=ax.transAxes)

graph_me = first_gen.loc[:, 'Total':'Speed']
graph_me
g = sns.pairplot(graph_me, palette="husl")
g.map_lower(corrfunc)
g.map_upper(corrfunc)
plt.show()
# list of my original dream team pokemons
dream_team = ['Pikachu', 'Eevee', 'Mew', 'Charmander', 'Squirtle', 'Gyarados']

# i will also include the 'evolved' forms of my dream team pokemons to show their potential strengths
# Mew - no further evolution
# Gyrados - no further evolution
# Eevee - Vaporeon, Jolteon, and Flareon
# Pikachu - (final evolution) Raichu
# Squirtle - (final evolution) Blastoise
# Charmander - (final evolution) Charizard 
dream_evolve = ['Vaporeon', 'Jolteon', 'Flareon', 'Raichu', 'Blastoise',
                'Charizard', 'CharizardMega Charizard X', 'CharizardMega Charizard Y', 
                'BlastoiseMega Blastoise']

for dream in dream_evolve:
    dream_team.append(dream)

first_gen.loc[first_gen['Name'].isin(dream_team), 'Team'] = True
team_list = first_gen.loc[first_gen.Name.isin(dream_team)]
team_list.sort_values(by=['Total'], ascending=False)  
# we want to compare how our pokemons are in comparison to the average stats
first_gen.describe() 
# i think it would make sense that we would want pokemons with Total (1) above mean and (2) above 75%.
first_gen.sort_values(by=['Total'], ascending=False).head(10) 