# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 

from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
videogames = pd.read_csv('/kaggle/input/videogamesales/vgsales.csv')
videogames
videogames.isnull().sum()
# Let's drop the null values 
videogames = videogames.dropna()
videogames.describe()
videogames.Year.value_counts().sort_index()
plt.figure(figsize=(12,4))
videogames.Year.value_counts().sort_index().plot(kind='bar')
# getting the each unique year in the dataframe videogames
uniqueYears = videogames.Year.unique()

# making a dictionary of dataframes to store each dataframe
dataFrameDict = {elem : pd.DataFrame for elem in uniqueYears}

# getting the dataframe for each entered year 
for key in dataFrameDict.keys():
    dataFrameDict[key] = videogames[:][videogames.Year == key]


# By doing this, we will be able to get the sales for each year and see how they compare across
# the US, Europe, and Japan
df1999 = dataFrameDict[1999]
df2016 = dataFrameDict[2016]
df2000 = dataFrameDict[2000]
df1 = pd.DataFrame({'x': ['North America', 'Europe', 'Japan', 'Other'], 'y': [df1999.NA_Sales.sum(), 
                                                                             df1999.EU_Sales.sum(), 
                                                                             df1999.JP_Sales.sum(), 
                                                                             df1999.Other_Sales.sum()]})
df2 = pd.DataFrame({'x':['North America', 'Europe', 'Japan', 'Other'], 'y': [df2016.NA_Sales.sum(), 
                                                                            df2016.EU_Sales.sum(), 
                                                                            df2016.JP_Sales.sum(),
                                                                            df2016.Other_Sales.sum()]})
df3 = pd.DataFrame({'x': ['North America', 'Europe', 'Japan', 'Other'], 'y': [df2000.NA_Sales.sum(), 
                                                                             df2000.EU_Sales.sum(), 
                                                                             df2000.JP_Sales.sum(), 
                                                                             df2000.Other_Sales.sum()]})
df1['hue']=1999
df2['hue']=2016
df3['hue']=2000
res=pd.concat([df1,df3, df2])
sns.barplot(x='x', y='y', data=res, hue='hue')
plt.title('Sales in 1999 and 2016')
plt.show()
videogames.Genre.unique()
df2008 = dataFrameDict[2008]
df2009 = dataFrameDict[2009]
df1 = pd.DataFrame({'x': ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 
                         'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy'], 
                   'y': [df2008.Genre.value_counts().sort_index()[i] for i in range(12)]})
df2 = pd.DataFrame({'x': ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 'Racing', 
                         'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy'], 
                   'y': [df2009.Genre.value_counts().sort_index()[i] for i in range(12)]})
df1['hue']=2008
df2['hue']=2009
res=pd.concat([df1,df2])
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='x', y='y', ax=ax, data=res, hue='hue')
plt.show()
df1999.Genre.value_counts().sort_index()
df2016.Genre.value_counts().sort_index()
df1 = pd.DataFrame({'x': ['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Puzzle', 
                         'Racing', 'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy'], 
                   'y': [df1999.Genre.value_counts().sort_index()[i] for i in range(12)]})
df2 = pd.DataFrame({'x':['Action', 'Adventure', 'Fighting', 'Misc', 'Platform', 'Racing',
                        'Role-Playing', 'Shooter', 'Simulation', 'Sports', 'Strategy'], 
                   'y': [df2016.Genre.value_counts().sort_index()[i] for i in range(11)]})
df1['hue']=1999
df2['hue']=2016
res=pd.concat([df1, df2])
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='x', y='y', ax=ax, data=res, hue='hue')
plt.show()
# we can use Counter to find the most common publishers 
publishers = Counter(videogames['Publisher'].tolist()).most_common(10)
labels = [i[0] for i in publishers]
counts = [i[1] for i in publishers]

fig,ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=labels, y=counts, ax=ax)
plt.xticks(rotation=90)
# We can now use counter for find the most common platforms
platforms = Counter(videogames['Platform'].tolist()).most_common(10)
labels=[i[0] for i in platforms]
counts = [i[1] for i in platforms]

fig,ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=labels, y=counts)
plt.xticks(rotation=90)