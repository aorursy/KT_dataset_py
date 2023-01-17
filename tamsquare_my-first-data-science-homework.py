# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import pylab
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2015.csv")
data.info()
data=data.drop(["Standard Error"],axis=1)
data.describe()
data.head(11)
data.Region.unique()
data.corr()
f,ax = plt.subplots(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True, fmt= '.2f',ax=ax,cmap="YlGnBu")
plt.show()
slope, intercept, r_value, p_value, std_err = stats.linregress(data.Freedom,data["Happiness Score"])
line = slope*data.Freedom+intercept
plt.figure(figsize=(15,10))
plt.plot(data.Freedom,data["Happiness Score"],"o",color="pink",alpha=0.7)
plt.plot(data.Freedom,line,alpha=1,color="gold",linewidth=2)
plt.ylabel("Happiness")
plt.xlabel("Freedom")
plt.legend()
pylab.title('Regression of Happiness Level Relation with Freedom')
ax = plt.gca()
fig = plt.gcf()
data.plot(kind="scatter",x="Family",y="Happiness Score",alpha=0.5,color="red",figsize=(15,10))
plt.show
plt.figure(figsize=(15,10))
data["Economy (GDP per Capita)"].plot(kind = 'line', color = 'red',label = 'Economy (GDP per Capita)',linewidth=1,alpha = 1,grid = True,linestyle = ':')
data["Health (Life Expectancy)"].plot(color = 'blue',label = 'Life Expectancy',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend()
plt.xlabel("Happiness Rank (0 is Best)")
plt.ylabel("Economy and Life Expectancy")
plt.title('Line Plot')            
plt.show()