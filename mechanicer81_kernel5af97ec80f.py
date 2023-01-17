#%% TOP TEN HAPPINESS COUNTRİES


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

data = pd.read_csv("../input/2017.csv")
#correlation
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

data.head(10)
data.columns
# Line Plot

data.Generosity.plot(kind = 'line', color = 'g',label = 'Happiness_Score',linewidth=1,alpha = 0.9,grid = True,linestyle = ':') # kind türüdür line, scatter, histogram gibi.
data.Freedom.plot(kind = 'line',color = 'r',label = 'Freedom',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.') # datamızın defense bilgilerini plot ettirmişizplt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='Happiness.Score', y='Freedom',alpha = 0.5,color = 'red') #here kind=scatter
plt.xlabel('Happiness')              # label = name of label
plt.ylabel('Freedom')
plt.title('Happiness - Freedom Scatter Plot') 
#%% Happiness of Countries
# Histogram

data.Family.plot(kind = 'hist',bins = 100,figsize = (12,12)) # happiness and freedom

plt.show()
#%% filterin freedom and health

data_frame2=data[np.logical_and(data['Freedom']>0.3, data['Health..Life.Expectancy.']>0.5)]
data_frame2.head(50)
data_frame3=data_frame2[np.logical_and(data_frame2['Generosity']>0.25, data_frame2['Trust..Government.Corruption.']>0.16)]
data_frame3.head(20)
#%%filterin family and economy

data_frame4=data_frame3[np.logical_and(data_frame3['Family']>1.2, data_frame3['Economy..GDP.per.Capita.']>1.3)]
data_frame4.head(20)
# Filtering top ten happiness countres
data_frame_last=data[(data['Freedom']>0.3) & (data['Health..Life.Expectancy.']>0.5) & (data['Trust..Government.Corruption.']>0.25) & (data['Dystopia.Residual']>1.4) & (data['Generosity']>0.3) & (data['Family']>1.4)] 
data_frame_last.columns
data_frame_last.head(10)

