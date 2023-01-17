import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns  

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv('../input/pokemon.csv')
data.info()
data.corr()


f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot



data.Speed.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')             

plt.ylabel('y axis')

plt.title('Line Plot')          

plt.show()
# Scatter Plot 



data.plot(kind='scatter', x='Attack', y='Defense',alpha = 0.5,color = 'red')

plt.xlabel('Attack')           

plt.ylabel('Defence')

plt.title('Attack Defense Scatter Plot')          
data.Speed.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()


data.Speed.plot(kind = 'hist',bins = 50)

plt.clf()

# Plotting all data 

data1 = data.loc[:,["Attack","Defense","Speed"]]

data1.plot()

# it is confusing
# subplots

data1.plot(subplots = True)

plt.show()
# scatter plot  

data1.plot(kind = "scatter",x="Attack",y = "Defense")

plt.show()
# hist plot  

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt