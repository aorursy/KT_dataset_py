# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2017.csv')
data2016 = pd.read_csv('../input/2016.csv')
data.info()
data.corr()
data.head(10)
f,ax = plt.subplots(figsize=(12, 10)) 
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) 
plt.show() 
data.columns
col_name =data.columns[1]  #fix for using
data=data.rename(columns = {col_name:'Happiness_Rank'})
col_name =data.columns[2]  #fix for using
data=data.rename(columns = {col_name:'Happiness_Score'})
col_name =data.columns[3]  #fix for using
data=data.rename(columns = {col_name:'Whisker_high'})
col_name =data.columns[4]  #fix for using
data=data.rename(columns = {col_name:'Whisker_low'})
col_name =data.columns[5]  #fix for using
data=data.rename(columns = {col_name:'Economy_GDP_per_Capita'})
col_name =data.columns[7]  #fix for using
data=data.rename(columns = {col_name:'Health_Life_Expectancy'})
col_name =data.columns[10]  #fix for using
data=data.rename(columns = {col_name:'Trust_Government_Corruption'})
col_name =data.columns[11]  #fix for using
data=data.rename(columns = {col_name:'Dystopia_Residual'})
data.columns
data.Health_Life_Expectancy.plot(kind='line', color ='r', label="Health Life Expectancy",linewidth=1,grid = True, linestyle = '-.', figsize = (16,10),title='Health Life Expectancy - Freedom - Generosity')
data.Freedom.plot(kind='line', color ='b', label="Freedom",linewidth=1,grid = True, linestyle = '-', figsize = (16,10))
data.Generosity.plot(kind='line', color ='g', label="Generosity",linewidth=1,grid = True, linestyle = '-', figsize = (16,10))
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.legend(loc='upper center')
plt.show()
series = data['Economy_GDP_per_Capita']        # data['Economy_GDP_per_Capita'] = series
print(type(series))
data_frame = data[['Economy_GDP_per_Capita']]  # data[['Economy_GDP_per_Capita']] = data frame
print(type(data_frame))
x = data['Economy_GDP_per_Capita']>1.5     # filtering
data[x]
data[np.logical_and(data['Economy_GDP_per_Capita']>1.5, data['Happiness_Rank']<20 )]
col_name =data2016.columns[1]  #fix for using
data2016=data.rename(columns = {col_name:'Happiness_Rank'})
col_name =data2016.columns[2]  #fix for using
data2016=data.rename(columns = {col_name:'Happiness_Score'})
col_name =data2016.columns[3]  #fix for using
data2016=data.rename(columns = {col_name:'Whisker_high'})
col_name =data2016.columns[4]  #fix for using
data2016=data.rename(columns = {col_name:'Whisker_low'})
col_name =data2016.columns[5]  #fix for using
data2016=data.rename(columns = {col_name:'Economy_GDP_per_Capita'})
col_name =data2016.columns[7]  #fix for using
data2016=data.rename(columns = {col_name:'Health_Life_Expectancy'})
col_name =data2016.columns[10]  #fix for using
data2016=data.rename(columns = {col_name:'Trust_Government_Corruption'})
col_name =data2016.columns[11]  #fix for using
data2016=data.rename(columns = {col_name:'Dystopia_Residual'})
plt.hist(data.Whisker_high,bins=50, label = 'Whisker high',)
plt.hist(data.Whisker_low,bins=50,alpha = 0.5, label = 'Whisker low') 
plt.legend(loc='upper right') 
plt.xlabel("Whisker high & low values")
plt.ylabel("frekans")
plt.title("Histogram")
plt.show()
plt.scatter(data.Happiness_Score,data.Economy_GDP_per_Capita, color="red",linewidths=0.1)
plt.title("data 2017 happines score & economoy gdp")




data.plot(grid=True,alpha=0.9,subplots=True, figsize= (12,12))

plt.subplot(2,1,1)
plt.plot(data.Happiness_Score ,data.Economy_GDP_per_Capita, color ="red")
plt.ylabel("'data'-2017 happines score & economoy gdp")

plt.subplot(2,1,2)
plt.plot(data2016.Happiness_Score ,data2016.Economy_GDP_per_Capita, color ="blue")
plt.ylabel("'data2016'-2016 happines score & economoy gdp")
plt.show()
