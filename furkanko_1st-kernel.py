import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns #visualization tool



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')     #This provides us to read data.
data.info()           #This gives us features of datas.
data.corr()                          #This gives us "correlation between features".
#correlation map

f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidth=1, fmt= '.2f', ax=ax)       #heatmap (Visualization function from "seaborn" library)

plt.show()
data.head(10)                   #This gives us first 10 Pokemons.
data.columns                     #This gives us columns seperately.
data.Speed.plot(kind='line', color='g', linewidth='3', label='Speed', alpha=0.5, grid=True, linestyle='-',figsize=(10,10))

data.Defense.plot(color='r', linewidth='3', legend='Defence', grid=True, linestyle=':')

plt.legend(loc='upper right')    #put label into plot

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind='scatter', x='Attack', y='Defense', label='Attack&Speed Relation', alpha=0.5, color='g', figsize=(10,10))

plt.xlabel('Attack')

plt.ylabel('Defense')

plt.legend(loc='upper left')

plt.title('Scatter Plot')

plt.show()

data.Speed.plot(kind='hist', bins=50, figsize=(10,10), color='b', grid=True)

plt.xlabel('Speed of Pokemons')

plt.ylabel('Number of Pokemons')

plt.show()
#plt.clf()                  We can not see plot due to clf(). (It clears.)