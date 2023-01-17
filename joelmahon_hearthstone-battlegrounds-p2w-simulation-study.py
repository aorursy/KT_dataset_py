import pandas as pd # data processing

import numpy as np

import random

import statistics as st

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

%matplotlib inline

import seaborn as sns

# game has 8 players, of which you are 1, assume you are always index 0

# each player has either 2 or 3 choices, the latter being signaled by a True value in the "P2W" column

# assume 24 total characters that are uniformly distributed with regards to their "power level" from 1 to 24 inclusive



character_count = 24

max_players = 8

p2w_count = 4

simu_count = 10000000 # total pick simulations



def pick_char(p2w, characters) :

    choices = []

    for _ in range(0, 3 if p2w else 2) :

        # take a character

        choices.append(random.choice(characters))

        # remove choice so players have no chance of getting the same option again on a further iteration

        characters.remove(choices[-1])

    

    # player chooses best from their options

    return max(choices)



def gen_game(p2w_count) :

    characters = [x for x in range(0, character_count)]

    # list of tuples (game index, p2w, character))

    players = []

    

    # add P2W players

    for _ in range(0, p2w_count) :

        players.append((True, pick_char(True, characters)))

        

    # add non-P2W players

    for _ in range(p2w_count, max_players) :

        players.append((False, pick_char(False, characters)))

        

    return players



def set_up_plot(data) :

    # +1 to account for upper not being used but the space being needed, and

    # another +1 to account for 1 indexing

    dist = sns.distplot(data, bins=np.arange(1,character_count+1+1) - 0.5, kde=False, hist_kws=dict(ec="k"))

    dist.xaxis.set_major_locator(ticker.MultipleLocator(1))



data = [[],[]]

    

for _ in range(0, simu_count // 2) :

    data[0].append(pick_char(True, [x for x in range(0, character_count)])+1) # +1 to remove 0 indexing

    data[1].append(pick_char(False, [x for x in range(0, character_count)])+1) # +1 to remove 0 indexing

    

############################### Histogram ###############################



# Set the width and height of the figure

plt.figure(figsize=(16,10))



# Add title

plt.title("Hearthstone Battlegrounds P2W Histogram")



# Chart               

set_up_plot(data[0])

set_up_plot(data[1])



# Add label for vertical axis

plt.xlabel("Character Rank")

plt.xlim(1, character_count + 1)



# Add label for vertical axis

plt.ylabel("Frequency")



plt.show()



############################### Boxplot ###############################



# Set the width and height of the figure

plt.figure(figsize=(10,10))



# Add title

plt.title("Hearthstone Battlegrounds P2W Boxplot")



# Chart

df0 = pd.DataFrame(data[0]).assign(P2W=True)

df1 = pd.DataFrame(data[1]).assign(P2W=False)



cdf = pd.concat([df0, df1])             # CONCATENATE

mdf = pd.melt(cdf, id_vars=['P2W'])     # MELT



sns.boxplot(x="P2W", y="value", data=mdf)  # RUN PLOT 



# Add label for vertical axis

plt.ylabel("Character Rank")



plt.show()



############################### Stats ###############################



def data_facts(data, name):

    print(name, "rank mean: ", st.mean(data))

    print(name, "rank std: ", st.stdev(data))

    

data_facts(data[0], "P2W")

data_facts(data[1], "Non-P2W")