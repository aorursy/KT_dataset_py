# INTRODUCTION



# I am saving the Consumer Financial Protection Bureau dataset for the first offical project.



# In lieu of this, I am using a 538 dataset on "battle royale", a game theory contest from Riddler Nation

# that has been run 3 times. I participated in the most recent round in May 2019. 

# Each round, a previous set of results was given to players.

# Find rules for the contest here: https://fivethirtyeight.com/features/are-you-the-best-warlord/



# During this exploration, I will explore how the introduction of past results has shifted gameplay.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import data

import pandas as pd

castle_solutions = pd.read_csv("../input/castle-solutions.csv")

castle_solutions_2 = pd.read_csv("../input/castle-solutions-2.csv")

castle_solutions_3 = pd.read_csv("../input/castle-solutions-3.csv")
# Rename columns with appropriate castle numbers (for future graphing)

castle_solutions = castle_solutions.rename(columns={"Castle 1": "Castle 01", 

                                 "Castle 2": "Castle 02",

                                 "Castle 3": "Castle 03",

                                 "Castle 4": "Castle 04",

                                 "Castle 5": "Castle 05",

                                 "Castle 6": "Castle 06",

                                 "Castle 7": "Castle 07",

                                 "Castle 8": "Castle 08",

                                 "Castle 9": "Castle 09"})



castle_solutions_2 = castle_solutions_2.rename(columns={"Castle 1": "Castle 01", 

                                 "Castle 2": "Castle 02",

                                 "Castle 3": "Castle 03",

                                 "Castle 4": "Castle 04",

                                 "Castle 5": "Castle 05",

                                 "Castle 6": "Castle 06",

                                 "Castle 7": "Castle 07",

                                 "Castle 8": "Castle 08",

                                 "Castle 9": "Castle 09"})



castle_solutions_3 = castle_solutions_3.rename(columns={"Castle 1": "Castle 01", 

                                 "Castle 2": "Castle 02",

                                 "Castle 3": "Castle 03",

                                 "Castle 4": "Castle 04",

                                 "Castle 5": "Castle 05",

                                 "Castle 6": "Castle 06",

                                 "Castle 7": "Castle 07",

                                 "Castle 8": "Castle 08",

                                 "Castle 9": "Castle 09"})
# Remove excess string data from datasets



# Remove verbatims

castles1 = castle_solutions.drop(['Why did you choose your troop deployment?'], axis=1)

castles2 = castle_solutions_2.drop(['Why did you choose your troop deployment?'], axis=1)

castles3 = castle_solutions_3.drop(['Why did you choose your troop deployment?'], axis=1)



# Show new dfs

castles1.head(10)
# Print out data 



# Check castle groupings

castles1.groupby('Castle 05').size()
# Plot troops sent to castles in battle 1

hist = castles1.hist(figsize=(20,16))
# Plot troops sent to castles in battle 2

hist = castles2.hist(figsize=(20,16))
# # UNCOMMENT IF YOU WISH TO RUN

# # This takes a long time to run (several minutes)



# # Overlays 3 battles' worth of histograms, by number of troops selected 



# # Print histogram of number of troops chosen per castle, over different battles

# bins = np.linspace(0, 100, 100)



# pyplot.hist(castles1['Castle 01'], bins, alpha=0.5, label='Battle 1')

# pyplot.hist(castles2['Castle 01'], bins, alpha=0.5, label='Battle 2')

# pyplot.hist(castles3['Castle 01'], bins, alpha=0.3, label='Battle 3')

# pyplot.legend(loc='upper right')

# pyplot.show()
# HOW MANY CASTLES ARE LEFT UNGUARDED, BY % PLAYERS PER BATTLE?



# Function to calculate % of unguarded castles for a given battle

# Returns a dictionary of castles with the % left unguarded



def unguarded_dict(df):

    # Calculate what percent of a castle for a given round was left ungaurded by players

    c1 = df['Castle 01'].value_counts(normalize=True)[0]

    c2 = df['Castle 02'].value_counts(normalize=True)[0]

    c3 = df['Castle 03'].value_counts(normalize=True)[0]

    c4 = df['Castle 04'].value_counts(normalize=True)[0]

    c5 = df['Castle 05'].value_counts(normalize=True)[0]

    c6 = df['Castle 06'].value_counts(normalize=True)[0]

    c7 = df['Castle 07'].value_counts(normalize=True)[0]

    c8 = df['Castle 08'].value_counts(normalize=True)[0]

    c9 = df['Castle 09'].value_counts(normalize=True)[0]

    c10 = df['Castle 10'].value_counts(normalize=True)[0]

    

    # Return dict with % left unguarded

    return {'Castle 01':c1, 'Castle 02':c2, 'Castle 03':c3, 'Castle 04':c4, 'Castle 05':c5,

            'Castle 06':c6, 'Castle 07':c7, 'Castle 08':c8, 'Castle 09':c9, 'Castle 10':c10}
# Set battles equal to dictionary of values

unguarded_battle_1 = unguarded_dict(castles1)

unguarded_battle_2 = unguarded_dict(castles2)

unguarded_battle_3 = unguarded_dict(castles3)
# Creating an empty dataframe with column names only

unguarded = pd.DataFrame()



# Append battle values to new df

unguarded = unguarded.append(unguarded_battle_1 , ignore_index=True)

unguarded = unguarded.append(unguarded_battle_2 , ignore_index=True)

unguarded = unguarded.append(unguarded_battle_3 , ignore_index=True)



# Rename index values to battle values

unguarded = unguarded.rename(index={0: 'Battle 1', 1: 'Battle 2', 2: 'Battle 3'})



# Check out df

unguarded
# Visual representation

# Plot heatmap of % of players who left castles unguarded, by castle and battle

import seaborn as sns

fig, ax = pyplot.subplots(figsize=(18,6)) 

sns.heatmap(unguarded, annot=True, vmin=0, vmax=0.4)
# Conclusions

# Based on the data reviewed and plotted, the data suggests that previous results do influence futures rounds of play.

# After battle #1, fewer players left Castle 8-10 unguarded. After battle #2, more players left Castle 1-3 unguarded.