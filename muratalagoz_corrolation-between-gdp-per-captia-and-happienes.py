# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/2017.csv')
data.head()
cdata = ['Country','Happiness_Rank','Happiness_Score','Whisker_high','Whisker_low','Economy_GDP_per_Capita','Family','Health_Life_Expectancy','Freedom','Generosity','Trust_Government_Corruption','Dystopia_Residual']
data.columns = cdata

data.info()
data.head()
plt.figure(figsize=(50,10))
sns.barplot(x=data.Country, y=data.Happiness_Score)
plt.xticks(rotation= 90)
plt.xlabel('Countries')
plt.ylabel('Happynes Score')
plt.title('Happyness Score of Given Countries')
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Economy_GDP_per_Capita.plot(kind = 'line', color = 'g',label = 'Economy_GDP_per_Capita',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Happiness_Score.plot(color = 'r',label = 'Happiness_Score',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='Economy_GDP_per_Capita', y='Happiness_Score',alpha = 0.5,color = 'red')
plt.xlabel('Economy_GDP_per_Capita')              # label = name of label
plt.ylabel('Happiness_Score')
plt.title('Economy_GDP_per_Capita Happiness_Score Scatter Plot')            # title = title of plot
f,ax1 = plt.subplots(figsize =(50,10))
sns.pointplot(x='Country',y='Economy_GDP_per_Capita',data=data,color='lime',alpha=0.8)
sns.pointplot(x='Country',y='Happiness_Score',data=data,color='red',alpha=0.8)
plt.text(40,0.6,'Economy_GDP_per_Capita',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'Generosity',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Countries',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Economy_GDP_per_Capita  VS  Happiness_Score',fontsize = 20,color='blue')
plt.grid()

f,ax1 = plt.subplots(figsize =(50,10))
sns.pointplot(x='Country',y='Economy_GDP_per_Capita',data=data,color='lime',alpha=0.8)
sns.pointplot(x='Country',y='Generosity',data=data,color='red',alpha=0.8)
plt.text(40,0.6,'Economy_GDP_per_Capita',color='red',fontsize = 17,style = 'italic')
plt.text(40,0.55,'Generosity',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Countries',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Economy_GDP_per_Capita  VS  Generosity',fontsize = 20,color='blue')
plt.grid()
g = sns.jointplot(data.Economy_GDP_per_Capita, data.Happiness_Score, kind="kde", size=7)
plt.savefig('graph.png')
plt.show()
sns.lmplot(x="Economy_GDP_per_Capita", y="Happiness_Score", data=data)
plt.show()
sns.kdeplot(data.Economy_GDP_per_Capita, data.Happiness_Score, shade=True, cut=3)
plt.show()
sns.pairplot(data)
plt.show()