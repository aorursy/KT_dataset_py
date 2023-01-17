# Let's import required libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visulization tool 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data_2015 = pd.read_csv("../input/world-happiness/2015.csv")
data_2015.head(5)
data_2015.tail(5)
data_2015.info()
data_2015.columns
# happiness score vs continents

f,ax = plt.subplots(figsize=(8,8))
sns.violinplot(data_2015['Happiness Score'], data_2015['Region'])

plt.show()
f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(data_2015.corr(), annot=True, linewidths = .5 ,fmt ='.1f', ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(5,5))
d = data_2015.loc[lambda data_2015: data_2015['Region'] == 'Western Europe']
sns.heatmap(d.corr(), annot=True, linewidths = .5 ,fmt ='.1f', ax=ax, cmap = 'Wistia')

plt.show()
f,ax = plt.subplots(figsize=(5,5))
d = data_2015.loc[lambda data_2015: data_2015['Region'] == 'North America']
sns.heatmap(d.corr(), annot=True, linewidths = .5 ,fmt ='.1f', ax=ax)

plt.show()
f,ax = plt.subplots(figsize=(5,5))
d = data_2015.loc[lambda data_2015: data_2015['Region'] == 'Middle East and Northern Africa']
sns.heatmap(d.corr(), annot=True, linewidths = .5 ,fmt ='.1f', ax=ax , cmap='rainbow')

plt.show()

f,ax = plt.subplots(figsize=(5,5))
d = data_2015.loc[lambda data_2015: data_2015['Region'] == 'Sub-Saharan Africa']
sns.heatmap(d.corr(), annot=True, linewidths = .5 ,fmt ='.1f', ax=ax , cmap='Blues')

plt.show()
f,ax = plt.subplots(figsize=(5,5))
d = data_2015.loc[lambda data_2015: data_2015['Region'] == 'Eastern Asia']
sns.heatmap(d.corr(), annot=True, linewidths = .5 ,fmt ='.1f', ax=ax ,)

plt.show()
# Line Plot

data_2015.Family.plot(kind = 'line', color = 'b', label = 'Family',linewidth=1, alpha = 1, grid = True, linestyle = ':')
data_2015.Freedom.plot(kind = 'line', color = 'g', label = 'Freedom',linewidth=1, alpha = 1, grid = True, linestyle = '-.')
data_2015.Generosity.plot(color = 'r',label = 'Generosity', linewidth=1, alpha = 1, grid = True, linestyle = '-.')
plt.legend(loc='upper right')    
plt.xlabel('Happiness Rank')            
plt.ylabel('0 - 1.5')
plt.title('Line Plot')  

plt.show()
# Scatter Plot
# x = Freedom , y = Generosity

data_2015.plot(kind = 'scatter', x = 'Freedom', y = 'Generosity',alpha = 1, color='red')
plt.xlabel('Freedom')
plt.ylabel('Generosity')
plt.title('Freedom & Generosity Scatter Plot')
plt.show()
# Histogram
data_2015.Freedom.plot(kind = 'hist',bins = 50,figsize = (8,8))
plt.show()