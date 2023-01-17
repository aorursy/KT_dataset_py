# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/military-expenditure-of-countries-19602019/Military Expenditure.csv')

#print(data.head())

# Any results you write to the current directory are saved as output.
col = ['Name','Type','1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974',

          '1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990',

           '1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006',

           '2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']

df = pd.DataFrame(data, columns = col)

#print(df.shape)

#print(df.describe)

#print(df['Type'].unique())

print(df.shape)

print(df.columns)
World = df[df['Name'] == 'World']

World = World.drop(['Type'], axis = 1)

World = World.set_index('Name')

World.index = World.index.rename('Year')

World = World.T

World = World[25:] #To only print the values of last 30 years.

World['World'] = World['World']/1e9

print(World.head(10))
plt.figure(figsize = (25,16))

World.plot(kind = 'bar', cmap = 'plasma')

plt.ylabel('Billions $', fontsize = 16)

plt.xlabel('Year', fontsize = 16)

plt.title('World Total Military Spending over Year', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
Top10 = (df[df['Type']== 'Country'].sort_values('2018', ascending = False).head(10))

print(Top10[['Name','2018']])
bottom10 = (df[df['Type']== 'Country'].sort_values('2018', ascending =True).head(10))

print(bottom10[['Name','2018']])
df1 = df.copy()

df1['diff'] = (df1[df1['Type'] == 'Country']['2018'] - df1[df1['Type'] == 'Country']['2017'])

df1['diff'] = df1['diff']/1e9

print(df1[['Name','diff']].sort_values('diff', ascending=False)[:10])
print(df1[['Name','diff']].sort_values('diff', ascending = True).head(10))
#Year 2017 Military Expenditure in Billion $

df1['2017_b'] = df1['2017']/1e9



#Percentage Change in Mititary Expediture over Last Year

df1['Perc'] = df1['diff']/df1['2017_b'] * 100 



#Top 10 Countries with Highest Change in Military Expenditure over last year.

print(df1[['Name','Perc']].set_index('Name').sort_values('Perc', ascending = False).head(10))
print(df1[['Name','Perc']].set_index('Name').sort_values('Perc', ascending = True).head(10))
print(df1[['Name','diff','Perc']].set_index('Name').sort_values('diff', ascending = False).head(10))
print(df1[['Name','diff','Perc']].set_index('Name').sort_values('Perc', ascending = True).head(10))
Nation = df[df['Type']== 'Country']

Nation = Nation.drop(['Type'], axis = 1)

Nation = Nation.set_index('Name')

Nation = Nation.T

Nation = Nation[25:]

print(Nation.head())
India = Nation['India']/1e9

China = Nation['China']/1e9

US = Nation['United States']/1e9

France = Nation['France']/1e9

Russia = Nation['Russian Federation']/1e9

UK = Nation['United Kingdom']/1e9

print(India.tail(5))

print(China.tail(5))

print(US.tail(5))

print(France.tail(5))

print(Russia.tail(5))

print(UK.tail(5))
plt.figure('figsize',(15,6))

plt.subplot(2,2,1)

plt.ylabel('Billion $')

plt.xlabel('Year')

plt.title('India Military Expediture')

India.plot(linestyle = '-', marker = '*', color = 'b')

plt.subplot(2,2,2)

plt.ylabel('Billion $')

plt.xlabel('Year')

plt.title('China Military Expediture')

China.plot(linestyle = '-', marker = '*', color = 'r')

plt.subplot(2,2,3)

plt.ylabel('Billion $')

plt.xlabel('Year')

plt.title('United States Military Expediture')

US.plot(linestyle = '-', marker = '*', color = 'y')

plt.subplot(2,2,4)

plt.ylabel('Billion $')

plt.xlabel('Year')

plt.title('France Military Expediture')

France.plot(linestyle = '-', marker = '*', color = 'b')

plt.xticks(rotation = 90)

plt.show()
plt.figure('figsize', (16,8))

India.plot(linestyle = '-', marker = '*', color = 'b')

China.plot(linestyle = '-', marker = '*', color = 'r')

US.plot(linestyle = '-', marker = '*', color = 'g')

France.plot(linestyle = '-', marker = '*', color = 'y')

Russia.plot(linestyle = '-', marker = '*', color = 'grey')

UK.plot(linestyle = '-', marker = '*', color = 'black')

plt.legend(['India', 'China', 'USA', 'France','Russia','UK'])

plt.xlabel('Years')

plt.ylabel('Billions $')

plt.title('Military Spending of 4 most Powerful Countries Today')

plt.grid(color='y', linestyle='-', linewidth=0.5)
plt.figure('figsize', (16,8))

India.plot(linestyle = '-', marker = '*', color = 'b')

France.plot(linestyle = '-', marker = '*', color = 'y')

Russia.plot(linestyle = '-', marker = '*', color = 'grey')

UK.plot(linestyle = '-', marker = '*', color = 'black')

plt.legend(['India', 'France','Russia','UK'])

plt.xlabel('Years')

plt.ylabel('Billions $')

plt.title('Who will be ahead among these 4 in the future')

plt.grid(color='y', linestyle='-', linewidth=0.5)