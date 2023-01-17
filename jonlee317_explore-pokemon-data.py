# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load all the fun data into variables :)

abilities = pd.read_csv("../input/abilities.csv")

items = pd.read_csv("../input/items.csv")

moves = pd.read_csv("../input/moves.csv")

movesets = pd.read_csv("../input/movesets.csv")

natures = pd.read_csv("../input/natures.csv")

pokemon = pd.read_csv("../input/pokemon.csv")

type_chart = pd.read_csv("../input/type-chart.csv")
# let's check out what abilities there are first



print(abilities)
# let's check out the items

print(items)
# let's skip over to see what kind of pokemon there are

pokemon.head()
# check out what type we have the most and least of

pokemon['type1'].value_counts().plot(kind="bar")
pokemon['hp'].value_counts().plot(kind="hist", bins=1000)
pokemon_weight = pokemon['weight'].str.replace(' lbs.', '').astype(float)

plt.scatter(pokemon['hp'], pokemon_weight)

plt.xlabel("Pokemon hp")

plt.ylabel("pokemon weight")



m, b = np.polyfit(pokemon['hp'], pokemon_weight, 1)

plt.plot(pokemon['hp'], m*pokemon['hp']+b)
type_chart.head()
type_chart.mean()
type_chart.mean().plot(kind="bar")
def check_def1_weakness(attack_type):

    plt.figure()

    (type_chart[type_chart[attack_type]>1])['defense-type1'].value_counts().plot(kind="bar", title="Weak Against "+attack_type)



def check_def1_strength(attack_type):

    plt.figure()

    (type_chart[type_chart[attack_type]<1])['defense-type1'].value_counts().plot(kind="bar", title="Strong Against "+attack_type)

    

def check_def2_weakness(attack_type):

    (type_chart[type_chart[attack_type]>=2])['defense-type2'].value_counts().plot(kind="bar", title="Weakness against "+attack_type)

    
# let's check the strength and weakness for each type



for poke in pokemon['type1'].unique():

    if poke.lower() != 'normal':

        check_def1_weakness(poke.lower())

        check_def1_strength(poke.lower())

    
movesets.head()
natures.head()
print(natures)
natures.mean(axis=1) 