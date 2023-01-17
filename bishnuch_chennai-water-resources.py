import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
df = pd.read_csv("../input/chennai-water-management/chennai_reservoir_levels.csv")

df.head()



df["Date"] = pd.to_datetime(df["Date"], format='%d-%m-%Y')

df.head()
df.info()
df.describe()
df.corr()
f,ax = plt.subplots(figsize=(4,4))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
df.CHOLAVARAM.plot(kind = 'line', color = 'r',label = 'CHOLAVARAM',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.CHEMBARAMBAKKAM.plot(color = 'b',label = 'CHEMBARAMBAKKAM',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('CHOLAVARAM AND CHEMBARAMBAKKAM PLOT')           

plt.show()
df.POONDI.plot(kind = 'line', color = 'r',label = 'POONDI',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.CHEMBARAMBAKKAM.plot(color = 'b',label = 'CHEMBARAMBAKKAM',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('POONDI AND CHEMBARAMBAKKAM PLOT')           

plt.show()
df.POONDI.plot(kind = 'line', color = 'r',label = 'POONDI',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

df.REDHILLS.plot(color = 'b',label = 'REDHILLS',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('POONDI AND REDHILLS PLOT')           

plt.show()
df.REDHILLS.plot(kind = 'hist',bins = 500,figsize = (10,10))

plt.show()
df.POONDI.plot(kind = 'hist',bins = 500,figsize = (10,10))

plt.show()
df.CHOLAVARAM.plot(kind = 'hist',bins = 500,figsize = (10,10))

plt.show()
sns.distplot(df['CHOLAVARAM'], color = 'red')

plt.title('Distribution of water levels of CHOLAVARAM', fontsize = 10)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
sns.distplot(df['REDHILLS'], color = 'red')

plt.title('Distribution of water levels of REDHILLS', fontsize = 10)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
sns.distplot(df['POONDI'], color = 'red')

plt.title('Distribution of water levels of POONDI', fontsize = 10)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()
sns.distplot(df['CHEMBARAMBAKKAM'], color = 'red')

plt.title('Distribution of water levels of CHEMBARAMBAKKAM', fontsize = 10)

plt.xlabel('Range of Year')

plt.ylabel('Count')

plt.show()