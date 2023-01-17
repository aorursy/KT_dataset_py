# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heroes_data = pd.read_csv("/kaggle/input/dota-heroes/p5_training_data.csv")
print(heroes_data.columns)
name_series = heroes_data['name']

print(name_series)
hero_count = len(name_series)

print("Hero Count: {}".format(hero_count))
print(heroes_data.loc[:,"name":"type"])
strength_filter = heroes_data["type"] == 0

agility_filter = heroes_data["type"] == 1

intelligence_filter = heroes_data["type"] == 2



strength_heroes = heroes_data[strength_filter]

agility_heroes = heroes_data[agility_filter]

intelligence_heroes = heroes_data[intelligence_filter]



# ----------------------------------------------

# A function that draws a horizontal line at the screen:

def line():

    print()

    print(100*"-")

    print()

# ----------------------------------------------



line()



print("STRENGTH HEROES")

print(strength_heroes.loc[:,"name":"type"].head())



line()



print("AGILITY HEROES")

print(agility_heroes.loc[:,"name":"type"].head())



line()



print("INTELLIGENCE HEROES")

print(intelligence_heroes.loc[:,"name":"type"].head())



line()
f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(heroes_data.corr() , annot = True , linewidths = .5 , fmt = '.1f' , ax = ax)

plt.show()
f,ax = plt.subplots(figsize = (18,18))

strength_heroes["maxDmg"].plot( kind = 'line' , color = 'red' , label = 'Strength Heroes' , linewidth = 2 , 

                                     alpha = 1 , grid = True , linestyle = '-' , ax=ax )

agility_heroes["maxDmg"].plot( kind = 'line' , color = 'green' , label = 'Agility Heroes' , linewidth = 2 , 

                                     alpha = 1 , grid = True , linestyle = '-' , ax=ax )

intelligence_heroes["maxDmg"].plot( kind = 'line' , color = 'blue' , label = 'Intelligence Heroes' , linewidth = 2 , 

                                     alpha = 1 , grid = True , linestyle = '-' , ax=ax )



plt.legend( loc = 'upper right' )

plt.xlabel('Heroes')

plt.ylabel('Maximum Damage')

plt.title('MAXIMUM DAMAGE OF HEROES')

plt.show()
filter = strength_heroes["maxDmg"] > 85

the_harsh_guy = strength_heroes[filter]

print(the_harsh_guy)
f,ax = plt.subplots(figsize = (10,10))

heroes_data.plot( kind = 'scatter' , x = 'baseStr' , y = 'moveSpeed' , alpha = 0.5 , color = 'red' , ax=ax )

plt.xlabel('Base Strength')

plt.ylabel('Move Speed')

plt.title('STRENGTH - SPEED  |  Scatter Plot')

plt.show()
strength_heroes["range"].plot( kind = 'hist' , bins = 50 , figsize = (8,8) )

plt.xlabel('Range')

plt.ylabel('Frequency')

plt.title('RANGE HISTOGRAM FOR STRENGTH HEROES')

plt.show()



agility_heroes["range"].plot( kind = 'hist' , bins = 50 , figsize = (8,8) )

plt.xlabel('Range')

plt.ylabel('Frequency')

plt.title('RANGE HISTOGRAM FOR AGILITY HEROES')

plt.show()



intelligence_heroes["range"].plot( kind = 'hist' , bins = 50 , figsize = (8,8) )

plt.xlabel('Range')

plt.ylabel('Frequency')

plt.title('RANGE HISTOGRAM FOR INTELLIGENCE HEROES')

plt.show()
f,ax = plt.subplots(figsize = (18,18))

sns.heatmap(heroes_data.corr() , annot = True , linewidths = .5 , fmt = '.1f' , ax = ax)

plt.show()