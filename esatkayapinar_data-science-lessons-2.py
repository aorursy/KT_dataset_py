import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns #Visulization library

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv") #Claming data from the dataset
data.info() #Showing some informations about the claimed dataset
data.corr() #Displaying a table about the relation between our variables 
f, ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt = '.4f', ax = ax)

plt.show()
data.head(10) #Gives us the first ten pokemons' information.
data.columns #Gives us the feauters(name of each cloumn)
#Parameters: kind = plot type, color = color, label = label, linewidth = width of each line, alpha = opacity, grid = grid, linestyle = style of line.

data.Speed.plot(kind = 'line', color = 'g', label = 'Speed', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':') #Graphic for speed feature

data.Defense.plot(kind = 'line', color = 'r', label = 'Defense', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.') #Graphic for defense feature

plt.legend(loc = 'upper right') #Defining the place of line guide

plt.xlabel('x axis') #Labeling x axis

plt.ylabel('y axis') #Labeling y axis

plt.title('Line Plot') #Naming title of graphic

plt.show()
#Parameters: kind = plot type, x = the column name in dataset, y = same like x, alpha = opacity, color = color.

data.plot(kind = 'scatter', x = "Attack", y = "Defense", alpha = 0.5, color = 'r')

plt.xlabel('Attack') #Labeling x axis

plt.ylabel('Defense') #Labeling y axis

plt.title('Scatter Plot') #Naming title of graphic

plt.show()
#Parameters: kind = type of plot, bins = number of bars, figsize = sizes of graphic.

data.Speed.plot(kind = 'hist', bins = 50, figsize = (12,12))

plt.xlabel('Speed') #Labeling x axis

plt.ylabel('Frequency') #Labeling y axis

plt.title('Histogram') #Naming title of graphic

plt.show()
plt.clf() #This function clears all of the plots and tables.