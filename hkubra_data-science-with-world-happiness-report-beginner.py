# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Read data from csv file
data17 = pd.read_csv('../input/2017.csv')
data16 = pd.read_csv('../input/2016.csv')
data15 = pd.read_csv('../input/2015.csv')
# Information about data
data15.info()
print("*******")
data16.info()
print("********")
data17.info()
print(data15.columns)
print(data16.columns)
print(data17.columns)
data15.rename(columns={"Happiness Rank" : "HapyRank",
                     "Happiness Score" : "HapyScr",
                     "Economy (GDP per Capita)" : "Economy",
                     "Health (Life Expectancy)" : "Health",
                     "Trust (Government Corruption)" : "Trust",
                     "Dystopia Residual" : "DystResd"}, inplace=True)
data15.drop(columns="Standard Error", inplace=True)
display(data15.head())
data16.rename(columns={"Happiness Rank" : "HapyRank",
                       "Happiness Score" : "HapyScr",
                       "Economy (GDP per Capita)" : "Economy",
                       "Health (Life Expectancy)" : "Health",
                       "Trust (Government Corruption)" : "Trust",
                       "Dystopia Residual" : "DystResd"}, inplace=True)
data16.drop(columns={"Lower Confidence Interval","Upper Confidence Interval"}, inplace=True)
display(data16.head())
data17.rename(columns={"Happiness.Rank" : "HapyRank",
                       "Happiness.Score" : "HapyScr",
                       "Economy..GDP.per.Capita." : "Economy",
                       "Health..Life.Expectancy." : "Health",
                       "Trust..Government.Corruption." : "Trust",
                       "Dystopia.Residual" : "DystResd"}, inplace=True)
data17.drop(columns={"Whisker.low","Whisker.high"}, inplace=True)
display(data17.head())
# Data Analysis
display(data15.describe())
display(data16.describe())
display(data17.describe())
# Data correlation
display(data15.corr())
display(data16.corr())
display(data17.corr())
# First 5 rows in 2017,2016,2015
display(data15.head())
display(data16.head())
display(data17.head())
# Last 10 rows 
display(data15.tail(10))
display(data15.tail(10))
display(data17.tail(10))
# First 3 countries and last 3 countries
print(data15['Country'][0:3])
print(data15['Country'][-3:])
print("****")
print(data16['Country'][0:3])
print(data16['Country'][-3:])
print("****")
print(data17['Country'][0:3])
print(data17['Country'][-3:])
# Correlation Map
f,ax=plt.subplots(figsize=(7,6))
sns.heatmap(data15.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)
f,ax=plt.subplots(figsize=(7,6))
sns.heatmap(data16.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)
f,ax=plt.subplots(figsize=(7,6))
sns.heatmap(data17.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)
plt.show()
# Line Plot
# Relations Between Economy and Trust 
data15.Economy.plot(kind = 'line', color = 'r',label='Economy15',linewidth=0.8 , grid = True,linestyle = '-',figsize = (6,6))
data15.Trust.plot(kind = 'line',color = 'g', label='Trust15',linewidth=0.7 , alpha = 0.9, grid = True,linestyle = '-',figsize = (6,6))
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.title(' 2015-Economy-Trust Line Plot')
plt.show()
data16.Economy.plot(kind = 'line', color = 'r',label='Economy16',linewidth=0.8 , grid = True,linestyle = '-',figsize = (6,6))
data16.Trust.plot(kind = 'line',color = 'g', label='Trust16',linewidth=0.7 , alpha = 0.9, grid = True,linestyle = '-',figsize = (6,6))
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.title('2016-Economy-Trust Line Plot')
plt.show()
data17.Economy.plot(kind = 'line', color = 'r',label='Economy17',linewidth=0.8 , grid = True,linestyle = '-',figsize = (6,6))
data17.Trust.plot(kind = 'line',color = 'g', label='Trust17',linewidth=0.7 , alpha = 0.9, grid = True,linestyle = '-',figsize = (6,6))
plt.ylabel('y axis')
plt.xlabel('x axis')
plt.title('2017-Economy-Trust Line Plot')
plt.show()


# Line Plot
# Countries' Health Life Expectancy with Happiness Score
data15.plot(subplots=True,x='HapyScr',y='Health', color='green',alpha=0.8,grid=True, figsize=(6,6))
plt.ylabel('Health')
plt.xlabel('Happiness Score')
plt.title('2015-Health-Happiness Line Plot')
data16.plot(subplots=True,x='HapyScr',y='Health', color='green',alpha=0.8,grid=True, figsize=(6,6))
plt.ylabel('Health')
plt.xlabel('Happiness Score')
plt.title('2016-Health-Happiness Line Plot')
data17.plot(subplots=True,x='HapyScr',y='Health', color='green',alpha=0.8,grid=True, figsize=(6,6))
plt.ylabel('Health')
plt.xlabel('Happiness Score')
plt.title('2017-Health-Happiness Line Plot')
plt.show()
# Scatter Plot
# Countries' Happiness Score and Economy by Years
plt.figure(figsize=(20,5))
plt.subplot(1,3,1)
plt.scatter(x=data15.Economy,y=data15.HapyScr, color='r',alpha=0.6)
plt.grid(True)
plt.xlabel('Economy')
plt.ylabel('Happiness Score')
plt.title("2015-Economy-Happiness Score Scatter Plot")
plt.subplot(1,3,2)
plt.scatter(x=data16.Economy,y=data16.HapyScr, color='r',alpha=0.6)
plt.grid(True)
plt.xlabel('Economy')
plt.ylabel('Happiness Score')
plt.title("Economy-Happiness Score Scatter Plot")
plt.subplot(1,3,3)
plt.scatter(x=data17.Economy,y=data17.HapyScr, color='r',alpha=0.6)
plt.grid(True)
plt.xlabel('Economy')
plt.ylabel('Happiness Score')
plt.title("Economy-Happiness Score Scatter Plot")
plt.show()
# Histogram
# Countries' freedom by Years
plt.figure(figsize=(20,5))
plt.subplot(131)
plt.hist(data15.Freedom ,color='red',bins = 50)
plt.title("2015-Countries' Freedom Histogram")
plt.subplot(132)
plt.hist(data16.Freedom ,color='red',bins = 50)
plt.title("2016-Countries' Freedom Histogram")
plt.subplot(133)
plt.hist(data17.Freedom ,color='red',bins = 50)
plt.title("2017-Countries' Freedom Histogram")
plt.show()
data15.Region.value_counts()
#data16.Region.value_counts()
#Pie Chart
labels=data15.Region.value_counts().index
explode = [0,0,0,0,0,0,0,0,0,0]
sizes = data15.Region.value_counts().values

labels2=data16.Region.value_counts().index
explode2 = [0,0,0,0,0,0,0,0,0,0]
sizes2 = data16.Region.value_counts().values
# visual
plt.figure(figsize = (18,7))
plt.subplot(131)
plt.pie(sizes, explode=explode, labels=labels, colors=sns.color_palette('Set2'), autopct='%1.2f%%')
plt.title('Region distribution in 2015',fontsize = 17,color = 'Brown')
plt.subplot(133)
plt.pie(sizes2, explode=explode2, labels=labels2, colors=sns.color_palette('Set3'), autopct='%1.2f%%')
plt.title('Region distribution in 2016',fontsize = 17,color = 'Green')
# First 10 rows and last 10 rows concatenating
# in 2015
dataFrame1=data15['Country']
dataFrame2=data15['HapyScr']
dataFrame3=data15['Freedom']
# axis=1 -> horizontal conc (columns)
dataHorizontal=pd.concat([dataFrame1,dataFrame2,dataFrame3],axis=1)
filtre1=dataHorizontal.head(10)
filtre2=dataHorizontal.tail(10)
# axis=0 ->vertical conc (rows)
newData=pd.concat([filtre1,filtre2],axis=0)
newData

# in 2016
dataFrame1=data16['Country']
dataFrame2=data16['HapyScr']
dataFrame3=data16['Freedom']
# axis=1 -> horizontal conc (columns)
dataHorizontal=pd.concat([dataFrame1,dataFrame2,dataFrame3],axis=1)
filtre1=dataHorizontal.head(10)
filtre2=dataHorizontal.tail(10)
# axis=0 ->vertical conc (rows)
newData=pd.concat([filtre1,filtre2],axis=0)
newData
# in 2017
dataFrame1=data17['Country']
dataFrame2=data17['HapyScr']
dataFrame3=data17['Freedom']
# axis=1 -> horizontal conc (columns)
dataHorizontal=pd.concat([dataFrame1,dataFrame2,dataFrame3],axis=1)
filtre1=dataHorizontal.head(10)
filtre2=dataHorizontal.tail(10)
# axis=0 ->vertical conc (rows)
newData=pd.concat([filtre1,filtre2],axis=0)
newData
# The 13 most happy countries in 2015,2016,2017
happy=data15['HapyScr']>7
display(data15[happy])
happy2=data16['HapyScr']>7
display(data16[happy2])
happy3=data17['HapyScr']>7
display(data17[happy3])
# Turkey's rank and happiness report in 2015,2016,2017
tr_data=data15[data15.Country == "Turkey"]
display(tr_data)
tr_data2=data16[data16.Country == "Turkey"]
display(tr_data2)
tr_data3=data17[data17.Country == "Turkey"]
display(tr_data3)
# The difference between Maximum's Happiness Score  and Turkey's Happiness Score in 2015,2016,2017
fark=data15['HapyScr'].max()-(tr_data['HapyScr'])
print(fark)
fark2=data16['HapyScr'].max()-(tr_data2['HapyScr'])
print(fark2)
fark3=data17['HapyScr'].max()-(tr_data3['HapyScr'])
print(fark3)
#Freedom and Happiness Score are Combine 
dataFree=data15['Freedom']>0.6
dataHappy=data15['HapyScr']< 6
display(data15[dataFree&dataHappy])

dataFree=data16['Freedom']>0.6
dataHappy=data16['HapyScr']< 6
display(data16[dataFree&dataHappy])

dataFree=data17['Freedom']>0.6
dataHappy=data17['HapyScr']< 6
display(data17[dataFree&dataHappy])
# Filtering pandas and logical_and
#  There are 9 countries.
data15[np.logical_and(data15.HapyScr>5.5, data15.Economy< data15.Economy.mean() )]

#  There are 9 countries.
data16[np.logical_and(data16.HapyScr>5.5, data16.Economy< data16.Economy.mean() )]
#  There are 7 countries.
data17[np.logical_and(data17.HapyScr>5.5, data17.Economy< data17.Economy.mean() )]
data15.loc[15:25,"Country":"Economy"]
data16.loc[15:25,"Country":"Economy"]
data17.loc[15:25,"Country":"Economy"]
dataFrame=data15['HapyRank'].tail(10)
for value in dataFrame:
    print("Value is ",value)
print('-The End-')
dataFrame=data16['HapyRank'].tail(10)
for value in dataFrame:
    print("Value is ",value)
print('-The End-')
dataFrame=data17['HapyRank'].tail(10)
for value in dataFrame:
    print("Value is ",value)
print('-The End-')