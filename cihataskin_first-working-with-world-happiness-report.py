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
data17=pd.read_csv('../input/2017.csv')
data16=pd.read_csv('../input/2016.csv')
data15=pd.read_csv('../input/2015.csv')
data17.info()
data17.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data17.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data17.head(10)
data17.tail(10)
data17.columns=[each.replace(".","")  
if(len(each.split("."))>1)else each for each in data17.columns]
data17.columns=[each.replace("(","")  
if(len(each.split("("))>1)else each for each in data17.columns]
data17.columns=[each.replace(")","")  
if(len(each.split(")"))>1)else each for each in data17.columns]

data17.columns
data16.columns=[each.replace(" ","")  
if(len(each.split(" "))>1)else each for each in data16.columns]
data16.columns=[each.replace("(","")  
if(len(each.split("("))>1)else each for each in data16.columns]
data16.columns=[each.replace(")","")  
if(len(each.split(")"))>1)else each for each in data16.columns]

data16.columns
data15.columns=[each.replace(" ","")  
if(len(each.split(" "))>1)else each for each in data15.columns]
data15.columns=[each.replace("(","")  
if(len(each.split("("))>1)else each for each in data15.columns]
data15.columns=[each.replace(")","")  
if(len(each.split(")"))>1)else each for each in data15.columns]

data15.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data17.Family.plot(kind = 'line', color = 'g',label = 'Family',linewidth=3,alpha = 0.5,grid = True,linestyle = ':')
data17.HappinessScore.plot(color = 'r',label = 'Happiness Score',linewidth=3, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Less Happy Country -->')              # label = name of label
plt.ylabel('Happiness')
plt.title('Line Plot')            
plt.show()
 # Scatter Plot 
# x = HappinessScore, y = EconomyGDPperCapita
data17.plot(kind='scatter', x='HappinessScore', y='EconomyGDPperCapita',alpha = 0.5,color = 'red')
plt.xlabel('HappinessScore')              # label = name of label
plt.ylabel('EconomyGDPperCapita')
plt.title('Happiness Score - Economy(GDPperCapita) Scatter Plot')            # title = title of plot
# Histogram
data17.Generosity.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data17.TrustGovernmentCorruption.plot(kind = 'line', color = 'g',label = '2017',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data16.TrustGovernmentCorruption.plot(color = 'r',label = '2016',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
data15.TrustGovernmentCorruption.plot(color = 'b',label = '2015',linewidth=1,grid = True,linestyle = '--')
plt.legend()    # legend = puts label into plot
plt.xlabel('Less Happy Country -->')              # label = name of label
plt.ylabel('Trust (Government Corruption)')
plt.title('Line Plot')            # title = title of plot
plt.show()




