import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
data=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')

data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15, 6))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(-5)
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.iyear.plot(kind = 'line', color = 'g',label = '',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.nkill.plot(color = 'r',label = 'nkill',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('sum attack')              # label = name of label
plt.ylabel('die people')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = year, y = kill
data.plot(kind='scatter', x='iyear', y='nkill',alpha = 0.5,color = 'red')
plt.xlabel('Year')              # label = name of label
plt.ylabel('Kill')
plt.title('Year Kill Scatter Plot')            # title = title of plot
plt.show()
# Histogram
# bins = number of bar in figure
data.iyear.plot(kind = 'hist',bins = 50,figsize = (10,10))
plt.show()
#create dictionary and look its keys and values
dictionary = {'Turkey' : 'TR','Afghanistan' : 'AFG'}
print(dictionary.keys())
print(dictionary.values())
x = data['nkill']>200     
data[x]
data[np.logical_and(data['nkill']>200, data['iyear']>2010 )]