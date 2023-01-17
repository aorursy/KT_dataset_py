import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Getting Data Ready
data_terror=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')

#Bring the first four rows from data_terror data
data_terror.head(4)
#Bring columns from data_terror data
data_terror.columns
#Change some column names in data_terror data
data_terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

#Bring columns from data_terror data
data_terror.columns
#Information about data_terror data
data_terror.info()
#Correlation between columns in data_terror
data_terror.corr()

#data_terror correlation map
f,ax = plt.subplots(figsize=(40, 40))
sns.heatmap(data_terror.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#Bring the first five rows from data_terror data
data_terror.head(5)
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data_terror.attacktype1.plot(kind = 'line', color = 'g',label = 'attacktype1',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data_terror.country.plot(color = 'r',label = 'country',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.xlabel('attacktype1')              # label = name of label
plt.ylabel('country')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = attacktype1, y = weaptype1
data_terror.plot(kind='scatter', x='attacktype1', y='country',alpha = 0.5,color = 'red')
plt.xlabel('attacktype1')              # label = name of label
plt.ylabel('country')
plt.title('attacktype1 country Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data_terror.attacktype1.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data_terror.attacktype1.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()