# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pokemon_data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv")
pokemon_data.info()
print(pokemon_data.corr())
f, ax = plt.subplots(figsize = (10,10)) # allcocating figure size

sns.heatmap(pokemon_data.corr(), annot = True, linewidths = .5, fmt = ".1f", ax=ax)

plt.show()
print(pokemon_data.head())
print(pokemon_data.columns)
plt.figure(2)

pokemon_data.Defense.plot(kind = "line", color = "orange", grid = True, linestyle = ":")

pokemon_data.Speed.plot(color = "green", grid = True, linestyle = ":")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Speed-Defense Line Plot")

plt.legend()

plt.show()
pokemon_data.plot(kind = "scatter", x = "Attack", y = "Defense", color = "orange")

plt.xlabel("Attack")

plt.ylabel("Defence")

plt.title("Attack vs Defense Scatter Plot")

plt.legend()

plt.show()
plt.figure(4)

pokemon_data.Speed.plot(kind = "hist", color = "green", bins = 80)

plt.xlabel("Speed")

plt.ylabel("Frequency")

plt.title("Pokemon Data Speed Visualization")

plt.show()
series = pokemon_data["Defense"]

print(type(series))
data_frame = pokemon_data[["Defense"]]

print(type(data_frame))
defense_bigger_200 = pokemon_data[pokemon_data.Defense > 200]

print(defense_bigger_200)
defense_bigger_200_and_attack_bigger_100 = pokemon_data[(pokemon_data["Defense"]>200) & (pokemon_data["Attack"]>100)]

# defense_bigger_200_and_attack_bigger_100 = pokemon_data[np.logical_and(pokemon_data["Defense"] > 200, pokemon_data["Attack"] > 100)]

print(defense_bigger_200_and_attack_bigger_100)
print(pokemon_data["Type 1"].value_counts(dropna = "False"))
print(pokemon_data["Type 2"].value_counts(dropna = "False"))
print(pokemon_data.describe())
pokemon_data.boxplot(column = "Attack", by = "Legendary")

plt.show()
pokemon_data.boxplot(column = "Speed", by = "Legendary")
pokemon_data.boxplot(column = "Speed", by = "Generation")
first_part = pokemon_data.head()

print(first_part)
melting_data = pd.melt(frame = first_part, id_vars = "Name", value_vars = ["Attack", "Speed"])
print(melting_data)
data_head = pokemon_data.head()

data_tail = pokemon_data.tail()
concatted_data_ver = pd.concat([data_head, data_tail], axis = 0)
print(concatted_data_ver)
concatted_data_hor = pd.concat([data_head,data_tail], axis = 1)

print(concatted_data_hor)
pokemon_data["Speed"] = pokemon_data["Speed"].astype("float")
print(pokemon_data["Type 2"].value_counts(dropna = False))

pokemon_data.dropna()

without_NaN = pokemon_data.dropna()
print(without_NaN)