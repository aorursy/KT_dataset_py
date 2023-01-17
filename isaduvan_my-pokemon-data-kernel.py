# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# import data

data =pd.read_csv("../input/pokemon.csv")

# I took sample for examine data

data.sample(10)
data.loc[63:63]
nan=data[data['name']=="NaN"]

nan
data=data.drop(['#'],axis=1) # drop column

# editing index number

data.index=range(1,801,1)
data.tail()
# editing columns name

data.rename(columns={"Name":"name"},inplace=True)

data.rename(columns={"Type 1":"type1"},inplace=True)

data.rename(columns={"Type 2":"type2"},inplace=True)

data.rename(columns={"HP":"hp"},inplace=True)

data.rename(columns={"Attack":"attack"},inplace=True)

data.rename(columns={"Defense":"defense"},inplace=True)

data.rename(columns={"Sp. Atk":"sp_atk"},inplace=True)

data.rename(columns={"Sp. Def":"sp_def"},inplace=True)

data.rename(columns={"Speed":"speed"},inplace=True)

data.rename(columns={"Generation":"generation"},inplace=True)

data.rename(columns={"Legendary":"legendary"},inplace=True)
data.columns
plt.figure(figsize=(8,6))

sns.barplot(x= data.type1.value_counts().index, y= data.type1.value_counts().values)

plt.xticks(rotation=45)

plt.grid()

plt.show()
plt.figure(figsize=(6,6))

sns.barplot(x= data.type2.value_counts().index, y= data.type2.value_counts().values)

plt.xticks(rotation=45)

plt.grid()

plt.show()
sort_legendary=data[data['legendary']==True] # filtred by legendary

plt.figure(figsize=(6,6))

sns.barplot(x= sort_legendary.type1.value_counts().index, y= sort_legendary.type1.value_counts().values)

plt.xticks(rotation=45)

plt.grid()

plt.show()



plt.figure(figsize=(6,6))

sns.barplot(x= sort_legendary.type2.value_counts().index, y= sort_legendary.type2.value_counts().values)

plt.xticks(rotation=45)

plt.grid()

plt.show()
filtre1 = sort_legendary.type1 =="Psychic"

filtre2=sort_legendary.type2=="Flying"

sort_legendary[filtre1 & filtre2]
filtre1 = sort_legendary.type1 =="Dragon"

filtre2=sort_legendary.type2=="Fire"

sort_legendary[filtre1 & filtre2]
non_legendary=data[data['legendary']==False]

plt.figure(figsize=(6,6))

sns.barplot(x= non_legendary.type1.value_counts().index, y= non_legendary.type1.value_counts().values)

plt.xticks(rotation=45)

plt.grid()

plt.show()
plt.figure(figsize=(6,6))

sns.barplot(x= non_legendary.type2.value_counts().index, y= non_legendary.type2.value_counts().values)

plt.xticks(rotation=60)

plt.grid()

plt.show()
filtre1 = non_legendary.type1 == "Water"

filtre2 = non_legendary.type2 == "Flying"

non_legendary[filtre1 & filtre2]
filtre1 = non_legendary.type1 == "Dragon"

filtre2 = non_legendary.type2 == "Flying"

non_legendary[filtre1 & filtre2]
non_legendary[filtre1]
plt.figure(figsize=(6,6))

sns.barplot(x= non_legendary.type1.value_counts().index, y= non_legendary.type1.value_counts().values, alpha=0.5)

sns.barplot(x= sort_legendary.type1.value_counts().index, y= sort_legendary.type1.value_counts().values)

plt.xticks(rotation=45)

plt.grid()

plt.show()
plt.figure(figsize=(6,6))

sns.barplot(x= non_legendary.type2.value_counts().index, y= non_legendary.type2.value_counts().values, alpha=0.5)

sns.barplot(x= sort_legendary.type2.value_counts().index, y= sort_legendary.type2.value_counts().values)

plt.xticks(rotation=45)

plt.grid()

plt.show()
sort_legendary.plot(kind="scatter",x="attack",y="defense")
sort_legendary.plot(kind="scatter",x="hp",y="defense",color="green")

sort_legendary.plot(kind="scatter",x="hp",y="attack",color="red")
sort_legendary.plot(kind="scatter", x="speed", y="attack")

sort_legendary.plot(kind="scatter", x="speed", y="defense")

sort_legendary.plot(kind="scatter", x="speed", y="hp")
data.plot(kind="scatter", x="defense", y="attack", color="green", grid=True)
