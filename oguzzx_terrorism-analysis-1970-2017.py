# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')
df.info()
df.describe()
df.columns
df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
df.columns
df = df[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]
df.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=2, fmt= '.1f',ax=ax)
plt.show()
df.head()
df.tail()
plt.figure(figsize=(10,10))
df.Year.plot(kind = 'line', color = 'b',label = 'Year',linewidth=2,alpha = 0.5,grid = True,linestyle = 'solid')
df.Killed.plot(kind = 'line', color = 'r',label ='Killed',linewidth=2,alpha = 0.5,grid = True,linestyle = 'solid')
plt.legend(loc="upper right")
plt.xlabel('x axis')            
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
df.plot(kind='scatter', x='Year', y='Killed',alpha = 0.5,color = 'red')
plt.xlabel('Year')              # label = name of label
plt.ylabel('Death')
plt.title('Year-Death Scatter Plot')         
df.Year.plot(kind="hist",bins=50,figsize=(12,12))
plt.show()
