import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data = pd.read_csv("../input/pokemon.csv")
data.info()

data.head(10)

data.columns
data.plot(kind='scatter', x='Attack', y='Defense', alpha=0.5, color='green')

plt.xlabel("Attack")

plt.ylabel("Defense")

plt.title("Attack vs Defense scatter plot")
data.Speed.plot(kind='hist', bins=50)
x = data['Defense']>200     

# There are only 3 pokemons who have higher defense value than 200

data[x]
data[(data['Defense']>200) & (data['Attack']>100)]
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = sum(data.Speed)/len(data.Speed)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]
# counting number of different pokemons

print(data['Type 1'].value_counts(dropna =False))  

# dropna is used so that if there are nan values that also be counted
# Melting

# Firstly I create new data from pokemons data to explain melt nore easily.

data_new = data.head()    # I only take 5 rows into new data

data_new

# id_vars = what we do not wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = 'Name', value_vars= ['Attack','Defense'])

melted