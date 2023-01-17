# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data5 = pd.read_csv('../input/2015.csv')

data6 = pd.read_csv('../input/2016.csv')

data7 = pd.read_csv('../input/2017.csv')
data5.info()

data6.info()

data7.info()

data5.columns



data5 = data5.set_index('Happiness Rank')  # We change index to happpiness score because

# happiness score work as index. 

data5.head()

data6 = data6.set_index('Happiness Rank')

data6.head()





data7= data7.set_index('Happiness.Rank')



data7.head()
#Line plot



data5['Happiness Score'].plot(kind='line',color='g',linewidth=2,alpha=0.7,grid=True, label = 'Happiness Score5', linestyle= ':')

data6['Happiness Score'].plot(kind='line',color='r',linewidth=2,alpha=0.7,grid=True,label = 'Happiness Score6',linestyle='-.')

plt.legend()

plt.xlabel('Happiness Score5')

plt.ylabel('Happness Score6')

plt.title('Line')

plt.show()

data5.Freedom.plot(kind='line',color='g',label='Freedom5',linewidth=2,alpha=0.7,grid=True)

data6.Freedom.plot(kind='line',color='blue',label='Freedom6',linewidth=2,alpha=0.7,grid=True)

plt.legend()

plt.xlabel('Freedom5')

plt.ylabel('Freedom6')

plt.title('Line')

plt.show()
data5.Family.plot(kind='line',color='g',label='Family5',linewidth=2,alpha=0.7,grid=True)

data7.Family.plot(kind='line',color='red',label='Family7',linewidth=2,alpha=0.7,grid=True)

plt.legend()

plt.xlabel('Family5')

plt.ylabel('Family7')

plt.title('Line')

plt.show()
data6['Dystopia Residual'].plot(kind='line',color='r',label='Dystopia Residual6',linewidth=2,alpha=0.7,grid=True)

data7['Dystopia.Residual'].plot(kind='line',color='g',label='Dystopia Residual7',linewidth=2,alpha=0.7,grid=True)

plt.legend()

plt.xlabel('Dystopia Residual6')

plt.ylabel('Dystopia Residual7')

plt.title('Line')

plt.show()



#Scatter plot



data5.plot(kind='scatter',x = 'Happiness Score', y = 'Freedom', alpha=0.7,grid=True,color='b')

plt.xlabel('Happiness Score')

plt.ylabel('Freedom')

plt.title('Scatter')

plt.show()

data6.plot(kind='scatter',x= 'Economy (GDP per Capita)',y ='Generosity',alpha=0.7,grid=True,color='r')

plt.xlabel('Economy (GDP per Capita)')

plt.ylabel('Generosity')

plt.title('Scatter')

plt.show()

data7.plot(kind='scatter', x = 'Health..Life.Expectancy.' , y = 'Family',alpha=0.7,grid=True,color='g')

plt.xlabel('HLE')

plt.ylabel('Family')

plt.title('Scatter')

plt.show()
#Histogram 



data5.Generosity.plot(kind='hist',figsize=(10,10),bins=50,alpha=0.7,color='r',grid=True)

plt.xlabel('Region')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()





data6.Freedom.plot(kind='hist',figsize=(10,10),bins=60,alpha=0.7,grid=True,color='b')

plt.xlabel('Freedom')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()
data7.Family.plot(kind='hist',figsize=(8,8),bins=40,color='g',alpha=0.7,grid=True)

plt.xlabel('Family')

plt.ylabel('Frequency')

plt.title('Histogram')

plt.show()



data5[(data5.Freedom > 0.5) & (data5.Generosity >0.5)] # There are 3.

data6[(data6.Freedom > 0.5) & (data6.Generosity >0.5)] # There are just 2.

data7[(data7.Freedom > 0.5) & (data7.Generosity >0.5)] # There are 4.



data5.head()
data5[(data5['Economy (GDP per Capita)'] > 1.3) & (data5['Trust (Government Corruption)'] > 0.4)] #There are 5.

data6[(data6['Economy (GDP per Capita)'] > 1.3) & (data6['Trust (Government Corruption)'] > 0.4)] #There are 7.

data7[(data7['Economy..GDP.per.Capita.'] > 1.3) & (data7['Trust..Government.Corruption.'] > 0.4)] #There are 3.
data5[(data5.Region == 'Western Europe') & (data5['Happiness Score'] > 7)] #8

data6[(data6.Region == 'Western Europe') & (data6['Happiness Score'] > 7)] #8



data5[(data5.Region == 'Middle East and Northern Africa') & (data5['Happiness Score'] > 7)] #1

data6[(data6.Region == 'Middle East and Northern Africa') & (data6['Happiness Score'] > 7)] #1
print(data5.Region.value_counts(dropna = False)) #statistics mostly from sub-saharanafrica and central and eastern europe.

print(data6.Region.value_counts(dropna= False))



data5.boxplot(column = 'Freedom', by = 'Region',figsize=(20,20))



data5.boxplot(column='Health (Life Expectancy)', by = 'Region', figsize=(15,15))



data5.boxplot(column = 'Family')



data5.boxplot(column = 'Health (Life Expectancy)')



data5.boxplot(column='Economy (GDP per Capita)', by ='Region', figsize=(15,15))



data6.head()
data6.boxplot(column = 'Freedom' , by = 'Region', figsize=(20,20))



data6.boxplot(column='Economy (GDP per Capita)', figsize = (15,15))



data6.boxplot(column = 'Lower Confidence Interval',figsize=(15,15))



data6.boxplot(column = 'Happiness Score' , by = 'Region', figsize = (15,15))



data6.boxplot(column= 'Freedom' , by = 'Region', figsize=(15,15))



data7.boxplot(column = 'Generosity')



data7.boxplot(column= 'Family', figsize=(15,15))



data7.boxplot(column='Trust..Government.Corruption.' , figsize=(15,15))



meanhap5 = sum(data5['Happiness Score'])/len(data5['Happiness Score'])

meanhap5

data5['Haappiness'] = ['Happy' if i > meanhap5 else 'Not Happy' for i in data5['Happiness Score']]



data5.head()



meaneco5 = sum(data5['Economy (GDP per Capita)'])/len(data5['Economy (GDP per Capita)'])

data5 ['Richness'] = ['Rich' if i > meaneco5 else 'Poor' for i in data5['Economy (GDP per Capita)']]

data5.head()



data5['Generosity2'] = data5['Economy (GDP per Capita)'] - data5['Generosity']

data5['Generostiy3'] = ['High' if i > 1 else 'Low ' for i in data5.Generosity2]

data5.head()
meanhap6 = sum(data6['Happiness Score'])/len(data6['Happiness Score'])

meanhap6

data6['Haappiness'] = ['Happy' if i > meanhap6 else 'Not Happy' for i in data6['Happiness Score']]



data6.head()





data8 = data5.groupby('Region').mean()

data9 = data6.groupby('Region').mean()



data8.plot(kind='line',color='r',label='Region5',linewidth=2,alpha=0.7,grid=True,linestyle=':')

data9.plot(kind='line',color='g',label='Region6',linewidth=2,alpha=0.7,grid=True,linestyle='-.')

plt.legend()

plt.xlabel('Region5')

plt.ylabel('Region6')

plt.title('Line')

plt.show()

x = data5.groupby('Region').Freedom.mean()

x



y = data6.groupby('Region').Freedom.mean()

y



data5.loc[0:10,'Country':'Family']



data6.loc[0:10, 'Country': 'Family']



data5.loc[0::10, 'Country':]



data6.loc[0::10,'Country':]



data7.loc[0::10,'Country':]