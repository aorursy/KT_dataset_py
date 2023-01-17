#import libraries

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns



#import files

#pokemon2 = pd.read_csv("../input/pokemon-index-edited/pokemon2.csv")

pokemon = pd.read_csv("../input/pokemon-challenge/pokemon.csv")

#tests = pd.read_csv("../input/pokemon-challenge/tests.csv")





# rename the column in pokemon data with "#" as "number" as its name

pokemon = pokemon.rename(index=str, columns={"#": "Number"})

pokemon
print("Dimenstions of Pokemon: " + str(pokemon.shape))
pokemon.isnull().sum()
combat = pd.read_csv("../input/pokemon-challenge/combats.csv")

print("Dimenstions of Combats: " + str(combat.shape))
combat
# calculate the win % of each pokemon 

# add the calculation to the pokemon dataset 

total_Wins = combat.Winner.value_counts()

#both methods produce the same results

countByFirst = combat.groupby('Second_pokemon').count()

countBySecond = combat.groupby('First_pokemon').count()

print("Looking at the dimensions of our dataframes")

print("Count by first winner shape: " + str(countByFirst.shape))

print("Count by second winner shape: " + str(countBySecond.shape))

print("Total Wins shape : " + str(total_Wins.shape))
# get the number of wins for each pokemon

numberOfWins = combat.groupby('Winner').count()

# Finding the total fights of each pokemon

numberOfWins['Total Fights'] = countByFirst.Winner + countBySecond.Winner

# Finding the win percentage of each pokemon

numberOfWins['Win Percentage']= numberOfWins.First_pokemon/numberOfWins['Total Fights']

print(numberOfWins)
# Merge the the original pokemon dataset with the winning dataset

results2 = pd.merge(pokemon, numberOfWins, right_index = True, left_on='Number')

results3 = pd.merge(pokemon, numberOfWins, left_on='Number', right_index = True, how='left')

results3

results3[np.isfinite(results3['Win Percentage'])].sort_values(by = ['Win Percentage'], ascending = False ).head(10)
results3[np.isfinite(results3['Win Percentage'])].sort_values(by = ['Win Percentage'], ascending = True ).head(10)
#plot graph of Speed vs Win Percentage

import matplotlib.pyplot as plt

sns.regplot(x="Speed", y="Win Percentage", data=results3, logistic=True).set_title("Speed vs Win Percentage")

#plot graph of Attack vs Win Percentage

sns.regplot(x="Attack", y="Win Percentage", data=results3, logistic=True).set_title("Attack vs Win Percentage")
#get the basic statistics of the data

results3.describe()
# 'Speed' is a continuous variable and we are going to use it as categorical variable to colour the parallel coordinates 

# so we need to divide it into range groups





#pd.plotting.parallel_coordinates(

    #pokemon[['Speed','HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def']], 'Speed')



#plt.show()
import matplotlib.pyplot as plt

from matplotlib import ticker

%matplotlib inline



import pandas as pd

import numpy as np

# 'Speed' is a continuous variable and we are going to use it as categorical variable to colour the parallel coordinates so we need to divide it into range groups

results3['Speed'] = pd.cut(results3['Speed'], [5, 45, 60, 150, 180])



cols = ['HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def','Win Percentage' ]

#cols = ['HP', 'Attack', 'Defense','Special Attack', 'Special Defense']



x = [i for i, _ in enumerate(cols)]

colours = ['Gold', 'Coral', 'MediumSeaGreen', 'MediumPurple']



# create dict of categories: colours

colours = {results3['Speed'].cat.categories[i]: colours[i] for i, _ in enumerate(results3['Speed'].cat.categories)}



# Create (X-1) sublots along x axis

fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))



# Get min, max and range for each column

# Normalize the data for each column

min_max_range = {}

for col in cols:

    min_max_range[col] = [results3[col].min(), results3[col].max(), np.ptp(results3[col])]

    results3[col] = np.true_divide(results3[col]- results3[col].min(), np.ptp(results3[col]))

    

results3 = results3.dropna()

# Plot each row

for i, ax in enumerate(axes):

    for idx in results3.index:

        Speed_category = results3.loc[idx,'Speed']

        ax.plot(x, results3.loc[idx, cols], colours[Speed_category])

    ax.set_xlim([x[i], x[i+1]])





# Set the tick positions and labels on y axis for each plot

# Tick positions based on normalised data

# Tick labels are based on original data

def set_ticks_for_axis(dim, ax, ticks):

    min_val, max_val, val_range = min_max_range[cols[dim]]

    step = val_range / float(ticks-1)

    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]

    norm_min = results3[cols[dim]].min()

    norm_range = np.ptp(results3[cols[dim]])

    norm_step = norm_range / float(ticks-1)

    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]

    ax.yaxis.set_ticks(ticks)

    ax.set_yticklabels(tick_labels)

    

for dim, ax in enumerate(axes):

    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))

    set_ticks_for_axis(dim, ax, ticks=6)

    ax.set_xticklabels([cols[dim]])

    

# Move the final axis' ticks to the right-hand side

ax = plt.twinx(axes[-1])

dim = len(axes)

ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))

set_ticks_for_axis(dim, ax, ticks=6)

ax.set_xticklabels([cols[-2], cols[-1]])





# Remove space between subplots

plt.subplots_adjust(wspace=0)



# Add legend to plot

plt.legend(

    [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in results3['Speed'].cat.categories],

    results3['Speed'].cat.categories,

    bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)



plt.title("Pokemon Index")



plt.show()
#filter according to "Type 1" and "HP"

#pokemon=pokemon[pokemon['Type 1']=='Grass'] 



#m1=results3['Win Percentage']> 0.90

#m2=results3['Win Percentage']<0.20

#results3=results3[m1] 



#sort top 10

#results3=results3.sort_values('Win Percentage')[:10]



#top 25%

totalrow=800

top25percent=int(800*0.25)

results3=results3.sort_values('Win Percentage')[:top25percent]
results3

#results3['Speed'] = pd.cut(results3['Speed'], [5, 45, 60, 150, 180])

cols = ['HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def','Win Percentage' ]



#cols = ['HP', 'Attack', 'Defense','Special Attack', 'Special Defense']



x = [i for i, _ in enumerate(cols)]

colours = ['Gold', 'Coral', 'MediumSeaGreen', 'MediumPurple']



# create dict of categories: colours

colours = {results3['Speed'].cat.categories[i]: colours[i] for i, _ in enumerate(results3['Speed'].cat.categories)}



# Create (X-1) sublots along x axis

fig, axes = plt.subplots(1, len(x)-1, sharey=False, figsize=(15,5))



# Get min, max and range for each column

# Normalize the data for each column

min_max_range = {}

for col in cols:

    min_max_range[col] = [results3[col].min(), results3[col].max(), np.ptp(results3[col])]

    results3[col] = np.true_divide(results3[col]- results3[col].min(), np.ptp(results3[col]))

    

results3 = results3.dropna()

# Plot each row

for i, ax in enumerate(axes):

    for idx in results3.index:

        Speed_category = results3.loc[idx,'Speed']

        ax.plot(x, results3.loc[idx, cols], colours[Speed_category])

    ax.set_xlim([x[i], x[i+1]])





# Set the tick positions and labels on y axis for each plot

# Tick positions based on normalised data

# Tick labels are based on original data

def set_ticks_for_axis(dim, ax, ticks):

    min_val, max_val, val_range = min_max_range[cols[dim]]

    step = val_range / float(ticks-1)

    tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]

    norm_min = results3[cols[dim]].min()

    norm_range = np.ptp(results3[cols[dim]])

    norm_step = norm_range / float(ticks-1)

    ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]

    ax.yaxis.set_ticks(ticks)

    ax.set_yticklabels(tick_labels)

    

for dim, ax in enumerate(axes):

    ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))

    set_ticks_for_axis(dim, ax, ticks=6)

    ax.set_xticklabels([cols[dim]])

    

# Move the final axis' ticks to the right-hand side

ax = plt.twinx(axes[-1])

dim = len(axes)

ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))

set_ticks_for_axis(dim, ax, ticks=6)

ax.set_xticklabels([cols[-2], cols[-1]])





# Remove space between subplots

plt.subplots_adjust(wspace=0)



# Add legend to plot

plt.legend(

    [plt.Line2D((0,1),(0,0), color=colours[cat]) for cat in results3['Speed'].cat.categories],

    results3['Speed'].cat.categories,

    bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)



plt.title("Pokemon Index")



plt.show()