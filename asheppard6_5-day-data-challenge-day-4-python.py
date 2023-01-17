# This kernel shows how to visualize catagorical variables



# Load in libraries 

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# List the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read into data frame

bites = pd.read_csv("../input/Health_AnimalBites.csv")



# Summarize data

#bites.describe()

bites.head()
# Plot counts of species as bar char

species = bites["SpeciesIDDesc"]

plot = sns.countplot(species).set_title("Counts of Species Bitten")