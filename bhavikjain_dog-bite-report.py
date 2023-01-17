import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/animal-bites/Health_AnimalBites.csv')
dataset.tail()
species = dataset.SpeciesIDDesc

species = species.dropna() 

speciesOfAnimal = species.unique()

print(speciesOfAnimal)
animal_list = []

for  i in speciesOfAnimal:

    animal_list.append(len(species[species==i]))

print(animal_list)
import seaborn as sns

count = dataset.BreedIDDesc.value_counts()

plt.figure(figsize=(15,8))

ax = sns.barplot(x=count[0:10].index,y=count[0:10])

plt.xticks(rotation=20)

plt.ylabel("Number of Bite")

plt.savefig('graph.png')

print(count[0:10].index)