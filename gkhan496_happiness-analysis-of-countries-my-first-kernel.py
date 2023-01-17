# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.listdir("../input")
plt.show()
# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2016.csv")
to_drop=['Lower Confidence Interval','Upper Confidence Interval','Dystopia Residual','Happiness Score']
data.drop(to_drop, inplace=True, axis=1)
new_columns={
    'Happiness Rank':'Happiness_Rank',
    'Health (Life Expectancy)':'Health',
    'Economy (GDP per Capita)':'Economy_GDP',
    'Trust (Government Corruption)':'Trust_Government'
}
data.rename(columns=new_columns, inplace=True)
dataH=data.head()
dataE=data.tail()
dataCon=pd.concat([dataH,dataE],axis=0,ignore_index=True)
dataCon
correlation  = data.corr()
correlation
f,ax = plt.subplots(figsize=(14, 14))
sns.heatmap(correlation,annot=True,linewidths=.5,vmin=-0.79,cmap="YlGnBu",vmax=1,linecolor="green",ax=ax)
plt.show()
data.plot(kind='scatter',x='Economy_GDP', 
           y='Family',alpha = 0.9,color = 'black',marker="x"
           ,grid=True,figsize=(7,7),fontsize=10,stacked=False)
plt.xlabel('Economy')
plt.ylabel('Family')
plt.title('Economy-Family')  
plt.show()

data.plot(kind='scatter',x='Happiness_Rank', 
           y='Family',alpha = 0.9,color = 'Purple',marker="H"
           ,grid=True,figsize=(7,7),fontsize=10,stacked=False)
plt.xlabel('Rank')
plt.ylabel('Family')
plt.title('Rank-Family')  
plt.show()
#Although economy and family have a direct impact.
#The effect of generosity and government trust to rank.
print("--Economy/Family--")
print("----------------------------------------------")
print("1-Denmark :","%.2f" %(data.loc[(0),'Economy_GDP']/data.loc[(0),'Family']))
print("82-China :","%.2f" %(data.loc[(82), 'Economy_GDP']/data.loc[(82),'Family']))
print("140-Angola :","%.2f" %(data.loc[(140),'Economy_GDP']/data.loc[(140),'Family']))
print("150-Syria :","%.2f" %(data.loc[(150),'Economy_GDP']/data.loc[(150),'Family']))
print("")
print(data.loc[[0],['Country','Trust_Government','Generosity']])
print("----------------------------------------------")
print(data.loc[[82],['Country','Trust_Government','Generosity']])
print("----------------------------------------------")
print(data.loc[[140],['Country','Trust_Government','Generosity']])
print("----------------------------------------------")
print(data.loc[[150],['Country','Trust_Government','Generosity']])
data.describe()
#The reason for the high values is the corruption in the world.
print(data.loc[[4],['Country','Generosity','Trust_Government']])
print(data.loc[[10],['Country','Generosity','Trust_Government']])
print(data.loc[[156],['Country','Generosity','Trust_Government']])
data.boxplot(column=["Generosity","Trust_Government"],fontsize=12,figsize=(10,10))
plt.show()