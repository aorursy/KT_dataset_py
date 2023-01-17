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


# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')

#First ten rows from data
data.head(10)
#Columns from data
data.columns
#The first letters of the titles are capitalized
data.columns=data.columns.str.capitalize()
#Information and describe in about data
data.info()
data.describe()
print(data.Country_txt.value_counts(dropna = False))
#Correlation between columns in data_terror
data.corr()
#data_terror correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
data.Success.plot(kind = 'line', color = 'g',label = 'Success',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Country.plot(color = 'r',label = 'Country',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.xlabel('Success')              
plt.ylabel('Country')
plt.title('Line Plot')            
plt.show()
#Attack types
data.Attacktype1.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data.Attacktype2.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
data.Attacktype3.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
plt.figure(figsize=(20,10))
data['Region'].plot(kind = 'line', color = 'b',label = 'Region',linewidth=2,alpha = 0.9,grid = True,linestyle = ':')
data['Nkill'].plot(color = 'r',label = 'Nkill',linewidth=2, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('Region')              # label = name of label
plt.ylabel('Kill')
plt.show()


# clf() = cleans it up again you can start a fresh
plt.clf()