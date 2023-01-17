# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data15=pd.read_csv('../input/2015.csv')
data16=pd.read_csv('../input/2016.csv')
data17=pd.read_csv('../input/2017.csv')
data15.info()
data17.describe()
data17.head(7)
data17.tail()
data17.corr()
#correlation map
f,x=plt.subplots(figsize=(12,12))
sns.heatmap(data17.corr(),annot=True,linewidths=.5, fmt= '.1f',ax=x)
plt.show()

data15.columns=['Country', 'Happiness_Rank', 'Happiness_Score', 'Whisker_high',
       'Whisker_low', 'Economy_GDP_per_Capita', 'Family',
       'Health_Life_Expectancy', 'Freedom', 'Generosity',
       'Trust_Government_Corruption', 'Dystopia_Residual']
data16.columns=['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',
       'Lower_Confidence', 'Upper_Confidence', 'Economy_GDP', 'Family',
       'Health_Life', 'Freedom', 'Trust_Government', 'Generosity',
       'Dystopia_Residual']
data17.columns=['Country', 'Happiness_Rank', 'Happiness_Score', 'Whisker_high',
       'Whisker_low', 'Economy_GDP_per_Capita', 'Family',
       'Health_Life_Expectancy', 'Freedom', 'Generosity',
       'Trust_Government_Corruption', 'Dystopia_Residual']

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data17.Happiness_Score.plot(kind = 'line', color = 'g',label = 'Happiness_Score',linewidth=1,alpha = 1,grid = True,linestyle = ':')
data17.Family.plot(kind='line',color = 'r',label = 'Trust_Government_Corruption',linewidth=1, alpha = 1,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attack, y = defense
data17.plot(kind='scatter', x='Happiness_Score', y='Health_Life_Expectancy',alpha = 0.5,color = 'red')
plt.xlabel('Happiness_Score')              # label = name of label
plt.ylabel('Health_Life_Expectancy')
plt.title('Happiness-Health Scatter Plot')
plt.show()
# Scatter Plot 
# x = attack, y = defense
data16.plot(kind='scatter', x='Happiness_Score', y='Economy_GDP',alpha = 0.5,color = 'red')
plt.xlabel('Happiness_Score')              # label = name of label
plt.ylabel('Economy_GDP')
plt.title('Happiness-Health Scatter Plot')
plt.show()
data16.columns
# Scatter Plot 
# x = attack, y = defense
data17.plot(kind='scatter', x='Happiness_Score', y='Family',alpha = 0.5,color = 'red')
plt.xlabel('Happiness_Score')              # label = name of label
plt.ylabel('Family')
plt.title('Happiness-Health Scatter Plot')
plt.show()
# Scatter Plot 
# x = attack, y = defense
data15.plot(kind='scatter', x='Happiness_Score', y='Dystopia_Residual',alpha = 0.5,color = 'red')
plt.xlabel('Happiness_Score')              # label = name of label
plt.ylabel('Dystopia_Residual')
plt.title('Happiness-Health Scatter Plot')
plt.show()
# Histogram
# bins = number of bar in figure
data17.Trust_Government_Corruption.plot(kind = 'hist',bins = 50,figsize = (10,7))
plt.show()
# Histogram
# bins = number of bar in figure
data17.Economy_GDP_per_Capita.plot(kind = 'hist',bins = 50,figsize = (10,7))
plt.show()
f, ax = plt.subplots(figsize=(20,5)) # set the size that you'd like (width, height)
plt.bar(data16.Region,data16.Happiness_Score)
plt.title("plot")
plt.xlabel("x")
plt.ylabel("y")
ax.legend(fontsize = 7)
plt.show()