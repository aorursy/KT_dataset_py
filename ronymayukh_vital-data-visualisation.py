# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#MAYUKH GHOSH 18BCE0417
import sklearn as sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import re
data = pd.read_csv("/kaggle/input/wildlife-strikes/database.csv")
attributes=(list(data))

attributes
species = data["Species Name"]

species_count=species.value_counts()

print(species_count)
species_count=species_count[species_count>4000]

print(species_count)
top_species = ["UNKNOWN MEDIUM BIRD","UNKNOWN SMALL BIRD","MOURNING DOVE", "GULL","UNKNOWN BIRD","KILLDEER", "AMERICAN KESTREL","BARN SWALLOW"]

top_species = species[species.isin(top_species)]

print(top_species.value_counts())
sns.countplot(top_species)

plt.title("Top Species That Impact with Aircraft")

plt.xticks(rotation='vertical')
top_known_species = ["MOURNING DOVE", "GULL","KILLDEER", "AMERICAN KESTREL","BARN SWALLOW"]

top_known_species = species[species.isin(top_known_species)]

print(top_known_species.value_counts())
sns.countplot(top_known_species)

plt.title("Top Known Species That Impact with Aircraft")

plt.xticks(rotation='vertical')
attributes=(list(data))

attributes
damage_x=[]

strike_x=[]

dam=".*Damage$"

stri=".*Strike$"

for i in attributes:

    if (re.match(dam, i)):

        damage_x.append(i)

    elif (re.match(stri, i)):

        strike_x.append(i)
damage_x
damage_x=damage_x[1:]

damage_x
strike_x
damage_y=[]

strike_y=[]

for i in strike_x:

    strike_y.append(data[i].sum())



for i in damage_x:

    damage_y.append(data[i].sum())
plt.bar(damage_x,damage_y)

plt.title("Parts Damaged in the Aircraft")

plt.xticks(rotation='vertical')
plt.bar(strike_x,strike_y,color='orange')

plt.title("Parts Striked in the Aircraft")

plt.xticks(rotation='vertical')
damage_per_strike=[]

parts=[]

for i in range(0,len(strike_x)):

    damage_per_strike.append(damage_y[i]/strike_y[i])

    parts.append(strike_x[i][:-7])
plt.bar(parts,damage_per_strike,color='red')

plt.title("Parts Damage per strike in the Aircraft")

plt.xticks(rotation='vertical')