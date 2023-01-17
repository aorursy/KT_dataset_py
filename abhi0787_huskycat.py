# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df=pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
df.isna().sum()
df.dropna(inplace=True)
df.isna().sum()
df['Name'].unique()
df['Genre'].unique()
df['Platform'].unique()
df.columns
#Analysing sales for different genres:

def analysingGenresOverTime(tempGenre,tempRegion):
    
    tempDates=[]
    tempSales=[]

    for i in range(len(df['Genre'])):
       
        if df['Genre'].iloc[i]==tempGenre:
            tempSales.append(df[tempRegion].iloc[i])
            tempDates.append(int(df['Year'].iloc[i]))
        
        
    zippedLists=zip(tempDates,tempSales)  
    sortedPairs=sorted(zippedLists)
    
    tuples = zip(*sortedPairs)
    list1, list2 = [ list(tuple) for tuple in  tuples]
    
    return list1,list2

#Analysing 'Racing' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Racing','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Racing Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Racing','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Racing Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Racing','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Racing Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#Analysing 'Action' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Action','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Action Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Action','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Action Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Action','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Action Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#Analysing 'Strategy' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Strategy','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Strategy Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Strategy','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Strategy Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Strategy','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Strategy Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#Analysing 'Strategy' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Puzzle','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Puzzle Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Puzzle','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Puzzle Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Puzzle','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Puzzle Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#Analysing 'Strategy' game Genre over differnet regions:


#North American sales:
list1,list2=analysingGenresOverTime('Shooter','NA_Sales')

import matplotlib.pyplot as plt
plt.plot(list1,list2)
plt.title('North American Sales for \'Shooter Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#European Sales: 
list1,list2=analysingGenresOverTime('Shooter','EU_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('European Sales for \'Shotter Game\' genre')
plt.ylabel('Sales in millions')
plt.show()

#Japan Sales: 
list1,list2=analysingGenresOverTime('Shooter','JP_Sales')

plt.figure()
plt.plot(list1,list2)
plt.title('Japan Sales for \'Shooter Game\' genre')
plt.ylabel('Sales in millions')
plt.show()


#Analysing North AMerican sales over genres: 
tempNames=df['NA_Sales'].groupby(df['Genre']).mean().keys()
tempLst=list(df['NA_Sales'].groupby(df['Genre']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Genre of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the North American region')
plt.show()


#Analysing Europeran sales over genres: 
tempNames=df['EU_Sales'].groupby(df['Genre']).mean().keys()
tempLst=list(df['EU_Sales'].groupby(df['Genre']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Genre of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the European region')
plt.show()


#Analysing Japan sales over genres: 
tempNames=df['JP_Sales'].groupby(df['Genre']).mean().keys()
tempLst=list(df['JP_Sales'].groupby(df['Genre']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Genre of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the Japan')
plt.show()
#Analysing sales over different platforms:

#Analysing North AMerican sales over genres: 
tempNames=df['NA_Sales'].groupby(df['Platform']).mean().keys()
tempLst=list(df['NA_Sales'].groupby(df['Platform']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Platform of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the North American region')
plt.show()


#Analysing Europeran sales over genres: 
tempNames=df['EU_Sales'].groupby(df['Platform']).mean().keys()
tempLst=list(df['EU_Sales'].groupby(df['Platform']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Platform of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the European region')
plt.show()


#Analysing Japan sales over genres: 
tempNames=df['JP_Sales'].groupby(df['Platform']).mean().keys()
tempLst=list(df['JP_Sales'].groupby(df['Platform']).mean())

plt.bar(tempNames,tempLst)
plt.tick_params(axis='x', rotation=70)
plt.xlabel('Platform of the games')
plt.ylabel('Sales in millions')
plt.title('Sales analysis over the Japan')
plt.show()
def topKGames(tempRegion):

    gameDict=dict()

    for i in range(len(df['Name'])):
        if df['Name'].iloc[i] not in gameDict.keys():
            gameDict.update({df['Name'].iloc[i]:tempRegion.iloc[i]})
        else:
            gameDict[df['Name'].iloc[i]]+=tempRegion.iloc[i]
            
    tempTupls=list(sorted(gameDict.items(),key=lambda x:x[1],reverse=True))
    
    tempNames=list(map(lambda temp:temp[0],tempTupls))
    tempVals=list(map(lambda temp:temp[1],tempTupls))
    
    return tempNames,tempVals
#Total Sales of top-5 games over the years for the NA region: 

tempNames,tempVals=topKGames(df['NA_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.xlabel('Top-5 games')
plt.title('Top-5 sold games in the North American region')
plt.show()
#Total Sales of top-5 games over the years for the European region: 

tempNames,tempVals=topKGames(df['EU_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.xlabel('Top-5 games')
plt.title('Top-5 sold games in the European region')
plt.show()
#Total Sales of top-5 games over the years for the Japanese region: 

tempNames,tempVals=topKGames(df['JP_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.xlabel('Top-5 games')
plt.title('Top-5 sold games in the Japanese region')
plt.show()
#top-k publishers:

def topKPublishers(tempRegion):

    gameDict=dict()

    for i in range(len(df['Publisher'])):
        if df['Publisher'].iloc[i] not in gameDict.keys():
            gameDict.update({df['Publisher'].iloc[i]:tempRegion.iloc[i]})
        else:
            gameDict[df['Publisher'].iloc[i]]+=tempRegion.iloc[i]
            
    tempTupls=list(sorted(gameDict.items(),key=lambda x:x[1],reverse=True))
    
    tempNames=list(map(lambda temp:temp[0],tempTupls))
    tempVals=list(map(lambda temp:temp[1],tempTupls))
    
    return tempNames,tempVals
#Total Sales of top-5 publishers over the years for the NA region: 

tempNames,tempVals=topKPublishers(df['NA_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.title('Top-5 publisheres in the North American region')
plt.show()
#Total Sales of top-5 publishers over the years for the European region: 

tempNames,tempVals=topKPublishers(df['EU_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.title('Top-5 publisheres in the European region')
plt.show()
#Total Sales of top-5 publishers over the years for the Japanese region: 

tempNames,tempVals=topKPublishers(df['JP_Sales'])

plt.bar(tempNames[:5],tempVals[:5])
plt.tick_params(axis='x', rotation=70)
plt.ylabel('Total sales in millions')
plt.title('Top-5 publisheres in the japanese region')
plt.show()
