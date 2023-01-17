# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/pokemon.csv")
data.head()
data.Speed.plot(kind = 'line', color = 'black', label = 'Speed', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')

data.Defense.plot(color = 'orange', label = 'Defense', linewidth = 1, alpha = 0.5, grid = True, linestyle ='-.')

plt.legend(loc='upper right') # it should be fixed for plotting

plt.xlabel('x axis')

plt.ylabel("y axis") # whether use "" or '' is doesn't change.

plt.title("Line plot")
data.columns
data.plot(kind ='scatter', x='Sp. Atk', y='Sp. Def', color = 'b', alpha = 0.5)

#plt.scatter(data.Attack, data.Defense, color = 'b', alpha = 0.5) # alternative way 

plt.xlabel("Special Attack")

plt.ylabel("Special Defense")

plt.title("Scatter plot for Sp")
# Distirubiton of Speed between pokemons

data.Speed.plot(kind = 'hist', bins = 30, figsize =(7,7))

plt.title("Histogram for Speed")
#dictionary example

dictionary = { 'spain' : 'madrid', 'usa' : 'new york' }
dictionary['spain'] = "barcelona"

dictionary['france'] = 'paris' # both "" and '' are usable for this area.Ü
print(dictionary)
del dictionary['spain']

print(dictionary)
dictionary.clear()

print(dictionary)
data_frame = data[['Defense']] # data_frame 

print(type(data_frame))

series = data['Defense']# series

print(type(series))
x = data['Defense'] > 150

data[x]
# if condition for python ?

data[np.logical_and(data['Defense'] > 200, data['Attack'] <100)] # filtering with logical_and

# data[(data['Defense'] > 200, data[])]
data_new = data.head()

data_new
# id_vars = what we dont wish to melt

# value_vars = what we want to melt

melted = pd.melt(frame=data_new,id_vars = "Name", value_vars = ["Attack","Defense"])

melted
melted.pivot(index = "Name", columns = "variable", values = "value")
data1 = data.head()

data2 = data.tail()



conc_data_row = pd.concat([data1,data2], axis = 0, ignore_index = True)

conc_data_row
data1 = data["Attack"].head()

data2 = data["Defense"].head()

conc_data_col = pd.concat([data1,data2], axis = 1)

conc_data_col

data.dtypes
data["Type 1"] = data["Type 1"].astype("category")

data.Speed = data.Speed.astype("float")

data.dtypes
data.info()
data["Type 2"].value_counts(dropna = False)
data["Type 2"].dropna(inplace = True)

data
assert data["Type 2"].notnull().all() # returns nothing cause its true
data = data.set_index("#")

data.head()
# indexing by using square brackets

#data["HP"][1]

data.HP[1]

# data[["HP","Attack"]]
data.HP[data.Speed < 15]
def div(n):

    return n/2

data.HP.apply(div)
data["total power"] = data.Attack + data.Defense

data.head()
data3 = data.copy()

data3.index = range(100,900,1)

data.head()
data1 = data.set_index(["Type 1","Type 2"])

data1.head(20)
# for pivoting

dic = {"treatment" : ["A","A","B","B"], "gender" : ["M","F","M","F"],"response" : [10,45,9,18], "age" : [15,62,32,43]}

df = pd.DataFrame(dic)

df
#pivoting

df.pivot(index = "treatment", columns = "gender", values = "response")