# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Health_AnimalBites.csv")
df.describe()
len(df)
df.head(10)
import matplotlib.pyplot as plt

%matplotlib inline
plt.hist(df["SpeciesIDDesc"])

plt.show()
df_dogs = df[df["SpeciesIDDesc"] == "DOG"]

print (len(df_dogs))

# df.groupby('BreedIDDesc').count()

df_dogs['BreedIDDesc'].value_counts().idxmax()
df["WhereBittenIDDesc"].value_counts()
# Get list of unique animals in the dataframe

list_of_animals = df.SpeciesIDDesc.unique()

# remove nan from list_of_animals

list_of_animals = list_of_animals[~pd.isnull(list_of_animals)]







# get rows filtered by a particular animal

def getAnimal(animal):

    return df[df["SpeciesIDDesc"] == animal]



# get the human body part most bitten by each animal

body_parts_bitten = np.array([[getAnimal(xx)["WhereBittenIDDesc"].value_counts().idxmax(), \

                     getAnimal(xx)["WhereBittenIDDesc"].value_counts().max()] \

                     for xx in list_of_animals])



print (list_of_animals)

print (body_parts_bitten)
plt.barh(list_of_animals, body_parts_bitten[:,1])

plt.show()
df["ResultsIDDesc"].value_counts()
rabies_animals = df[df["ResultsIDDesc"] == "POSITIVE"]

rabies_animals