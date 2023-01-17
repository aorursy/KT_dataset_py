import numpy as np

import pandas as pd

from operator import itemgetter

import matplotlib.pyplot as plt



df = pd.read_csv('../input/Pokemon.csv')

df.head()
def fight(pokemon1, pokemon2):

    '''

    Input:

    Two names of the fighting Pokemon

    

    Output:

    A list of the winner/s

    

    Annotation:

    There is a small chance for an attack to missfire (p=0.1).

    There are no weaknesses or strengths against types implemented.

    Certainly this is not the real battle-algorithm.

    '''

    # Retrieving every needed value for the fight

    hp_1, atk_1, def_1 = df[df['Name']==pokemon1][['HP', 'Attack', 'Defense']].values[0]

    hp_2, atk_2, def_2 = df[df['Name']==pokemon2][['HP', 'Attack', 'Defense']].values[0]

    def_max = df['Defense'].max()

    

    # Normalize the defense to use it as probability

    def_1 /= def_max

    def_2 /= def_max

    

    # Iterate until someone loses

    while (hp_1>0) & (hp_2>0):

        # Choose randomly whether the attack misses

        missed = np.random.choice([1, 0], 2, p=[0.9, 0.1])

        

        # Compute the attack of pokemon1

        # If added a factor to slow the fight down a bit

        hp_2 -= 0.25* (1-def_2) * atk_1 * missed[0]

        hp_1 -= 0.25* (1-def_1 )* atk_2 * missed[1]

        

    # Check for a winner

    if hp_1<=0 and hp_2<=0:

        return [pokemon1, pokemon2]

    elif hp_1<=0:

        return [pokemon2]

    else:

        return [pokemon1]
fights = 10000



winner_dict = {}

for i in range(fights):

    fighter = df['Name'].sample(2).values

    result = fight(fighter[0], fighter[1])

    for winner in result:

        if winner in winner_dict.keys():

            winner_dict[winner] += 1

        else:

            winner_dict[winner] = 1
top = 30



winner_list = sorted(list(winner_dict.items()), key=itemgetter(1), reverse=False)

w_name, w_wins = zip(*winner_list)



index = np.arange(top)

plt.figure(figsize=(10,8))

plt.barh(index, w_wins[-top:])

plt.xlabel('Number of wins')

plt.ylabel('Pokemon')

plt.title('What are the best Pokemon of our arena?')

plt.yticks(index, w_name)

plt.show()