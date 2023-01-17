import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sbn

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3



# Input data files are available in the "../input/" directory. 

foodfacts=pd.read_csv("../input/FoodFacts.csv")

foodfacts.head()#print the data
ax=sbn.boxplot(x="energy_100g",y="sugars_100g",data=foodfacts)

ax=sbn.stripplot(x="energy_100g",y="sugars_100g",data=foodfacts,jitter=True)
foodfacts.plot(kind="scatter",x="energy_100g", y="sugars_100g",title="A plot of energy to sugar", legend=True)
sbn.jointplot(x="energy_100g", y="sugars_100g", data=foodfacts, size=5)


ax = sbn.stripplot(x="energy_100g", y="sugars_100g", data=foodfacts, jitter=True, edgecolor="gray")

ax=sbn.boxplot(x="energy_100g", y="sugars_100g", data=foodfacts)
sbn.violinplot(x="energy_100g", y="sugars_100g", data=foodfacts, size=5)