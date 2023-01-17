# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns


data2015 = pd.read_csv('../input/2015.csv')
data2016 = pd.read_csv('../input/2016.csv')
data2017 = pd.read_csv('../input/2017.csv')
print(data2015.head(10))


data2015.info()
data2015.corr()
#correlation map
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data2015.corr(),annot=True, linewidths=.5, fmt='.1f' , ax=ax)
plt.show()

#Lets change column names which are problem while codding on python
data2015.rename(columns ={'Happiness Score':'Happiness_Score', 'Economy (GDP per Capita)': 'Economy' , 'Health (Life Expectancy)':'Health', 'Happiness Rank':'Happiness_Rank'},inplace=True)
#Line plot

data2015.columns

data2015.Economy.plot(kind = 'line', color = 'blue',label = 'Economy',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')
data2015.Happiness_Score.plot(color = 'red',label = 'Happiness_Score',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()

#Scatter Plot
#x = Economy, y = Happiness_Score
data2015.plot(kind='scatter', x='Economy', y='Happiness_Score', alpha=0.5,color='blue')
plt.xlabel('Economy')
plt.ylabel('Happiness_Score')
plt.title('Economy-Happiness_Score Scatter Plot')
plt.show()

#histogram
data2015.Economy.plot(kind='hist',bins=40,figsize=(10,10))
plt.show()
#sorting by economy
data2015.sort_values(['Economy','Country'])
#Filtering by Happiness_Rank
data2015[data2015.Happiness_Rank <=20]
#Top 20 Happy Country sorting by Economy
data2015[data2015.Happiness_Rank<=20].sort_values('Economy',ascending=False)
data2015[(data2015.Region=='Western Europe')&(data2015.Freedom<=1)]
#
print(data2015.groupby('Region').Economy.mean().sort_values(ascending=False))
data2015.groupby('Region').Economy.mean().sort_values(ascending=False).plot(kind='bar')
plt.show()
#Aggregation
data2015.groupby('Region').Economy.agg(['mean','max','min','count','median']).sort_values('mean',ascending='False')
#Lets add new column and change the Country name with upper letters
data2016['Country_Upper_Letters']=data2016.Country.str.upper()
data2016.head(5)

#drop column
data2016=data2016.drop(['Country_Upper_Letters'], axis=1)
#Who contains Europe string, that is to say lets see all Europe
data2015[data2015.Region.str.contains('Europe')]
#replace, #lets change space characters(' ') with underscore character('_').
data2016.columns.str.replace(" ",'_')

newdata=data2015.head(5)
melted_data=pd.melt(frame=newdata, id_vars='Country', value_vars=['Happiness_Rank','Economy','Family'])
melted_data
#pivoting - reverse of melting
melted_data.pivot(index='Country', columns='variable',values='value')
#concatenating data
eco2015=data2015['Economy'].head(20)
eco2017=data2017['Economy..GDP.per.Capita.'].head(20)
alleco=pd.concat([eco2015,eco2017],axis=1)
alleco
