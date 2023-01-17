



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import os

print(os.listdir("../input"))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
pokedata = pd.read_csv('../input/Pokemon.csv')
pokedata.info() 

pokedata.head() 
pokedata.columns
print('Max Total: '+pokedata.loc[pokedata['Total'].idxmax()].Name+' / '+'TotalValue: '+str(pokedata['Total'].max()))
print('Max Defense: '+pokedata.loc[pokedata['Defense'].idxmax()].Name+' / '+'DefenseValue: '+str(pokedata['Defense'].max()))
print('Max Attack: '+pokedata.loc[pokedata['Attack'].idxmax()].Name+' / '+'AttackValue: '+str(pokedata['Attack'].max()))
print('Min Total: '+pokedata.loc[pokedata['Total'].idxmin()].Name+' / '+'TotalValue: '+str(pokedata['Total'].min()))
pokedata['Type 1'].value_counts() #Pokemon Types
pokedata.corr()
f,ax = plt.subplots(figsize=(18,18))

sns.heatmap(pokedata.corr(), annot=True, linewidths= .5, fmt= '.1f',ax=ax)

plt.show()
pokedata.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

pokedata.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

pokedata.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')              # label = name of label

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

pokedata.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()