import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

infile = '../input/Health_AnimalBites.csv'
df = pd.read_csv(infile)

df.head()
d2 = pd.concat([df[df.SpeciesIDDesc == 'CAT'], df[df.SpeciesIDDesc == 'DOG']]).reset_index(drop=True)
d2.head()
n = d2["SpeciesIDDesc"].value_counts() / len(df) * 100

n
n.plot(kind="bar")

plt.yticks(np.arange(0,101,10))

plt.title("Dog and Cat Bite Reports")

plt.ylabel("Percentage of Bites")