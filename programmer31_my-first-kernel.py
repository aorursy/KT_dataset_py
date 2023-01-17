# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importing data
data = pd.read_csv("../input/Pokemon.csv")
#Basic info about the data
data.info()
data.columns
#A bir more complicated information about the numerical data
data.describe()
#The first N data (here N = 10)
data.head(10)
#The corrolation between the data
data.corr()
 #Line plot showing the attack and defense values of pokemons
data.Attack.plot(kind = "line", label = "attack", color = "red", figsize = (15, 15), linestyle = ":", linewidth = 1.5, grid = True)
data.Defense.plot(kind = "line", label = "defense", color = "blue", figsize = (15, 15), linestyle = "-", linewidth = 1.5, alpha = 0.5, grid = True)
plt.legend(loc = "upper left")
plt.xlabel("index")
plt.ylabel("y label")
plt.show()
#Scatter plot showing how attack correlates with special attack 
data.plot(kind = "scatter", x = "Attack", y = "Sp. Atk", color = "blue", figsize = (14, 14), grid = True)
plt.show()
#Histogram showing the frequency of health values
data.HP.plot(kind = "hist", bins = 75, figsize = (15, 15))
plt.xlabel("HP")
plt.show()
#Create a data frame
df = data[["HP"]]
#Set a filter
x = data["HP"] < 40
#Print the filtered data
#If your write df[x] instead you will only see the hp values
data[x]
#Filtering using logical_and
data[np.logical_and(data["HP"] > 100, data["Attack"] > 100)]
#Another way of using "and"
data[(data["HP"] > 100) & (data["Attack"] > 100)]