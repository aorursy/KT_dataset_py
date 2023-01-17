import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Read the file into a dataset

bites = pd.read_csv("../input/Health_AnimalBites.csv")



# See first few rows

bites.head()



# Pick column with a categorical variable in it = species

species = bites["SpeciesIDDesc"]



# Plot a bar chart with species info

sns.countplot(species)

plt.title("Data on animal bites by species")

plt.xlabel("Species")



## Dogs are associated with the largest number of bites by far, probably because they are

## the most common pets, not necessarily because they are more dangerous.
# Plot a pie chart with animal gender id



# Start by counting the amount of bites by gender

genderCount = bites["GenderIDDesc"].value_counts()

print(genderCount)



# Now plot this in a pie chart

labels = 'Male', 'Female', 'Unknown'

counts = [3832, 2016, 629]

colors = ['yellowgreen', 'lightcoral', 'lightskyblue']



plt.pie(counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)

plt.axis('equal')

plt.title("Percentage of bites by animal gender")

plt.show()



## Males are biters!