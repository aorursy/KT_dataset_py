#import libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



#read our file

pokemon = pd.read_csv("../input/Pokemon.csv")

print(pokemon)

#look only at numeric values

pokemon.describe()

#this line of code descibes the non numeric values too

#pokemon.describe(include="all")

#List all column names

print(pokemon.columns)



#get the Legendary type

leg = pokemon["Legendary"]



plt.hist(leg)



hp = pokemon["HP"]

plt.hist(hp,bins = 9, edgecolor = "black")

plt.title("HP of all the pokemon in the dataset")

#from the graph we can see that the pokemon from 50 to 75 have the maximum HP

plt.xlabel("HP of the pokemon")

plt.ylabel("count")