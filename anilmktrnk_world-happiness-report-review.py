# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns #Visulation Tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data15 = pd.read_csv('../input/2015.csv') # Data Read From CSV
data16 = pd.read_csv('../input/2016.csv')
data17 = pd.read_csv('../input/2017.csv')
data15.info() #Data info
data15.corr()  # Let's look at the relationship between columns.
f,ax = plt.subplots(figsize=(25,25))  #Map size.
sns.heatmap(data15.corr(), annot=True,linewidths=.5,fmt='.2f',ax=ax) 
# anot=Numbers Appearence,fmt=Digits After a Comma
plt.show()
data15.head(10)  # Top 10 line display 
data15.columns #Columns display
data15 = data15.rename(index=str, columns={"Happiness Rank": "Happiness_Rank", "Happiness Score": "Happiness_Score", "Standard Error": "Standard_Error", "Economy (GDP per Capita)": "Economy_GDP_per_Capita", "Dystopia Residual": "Dystopia_Residual","Trust (Government Corruption)": "Trust_Government_Corruption",'Health (Life Expectancy)': "Health_Life_Expectancy",})
data15.columns
# Line Plot 

data15.Happiness_Score.plot(kind='line', color='r',label='Happiness Score', linewidth=2,alpha=1,grid=True,linestyle='-',figsize=(20, 20))
data15.Economy_GDP_per_Capita.plot(kind='line', color='b',label='Economy GDP Per Capita', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Family.plot(kind='line', color='g',label='Family', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Health_Life_Expectancy.plot(kind='line', color='black',label='Health_Life_Expectancy', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Freedom.plot(kind='line', color='orange',label='Freedom', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Trust_Government_Corruption.plot(kind='line', color='brown',label='Trust_Government_Corruption', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Generosity.plot(kind='line', color='gray',label='Generosity', linewidth=2,alpha=1,grid=True,linestyle='-')
data15.Dystopia_Residual.plot(kind='line', color='pink',label='Dystopia_Residual', linewidth=2,alpha=1,grid=True,linestyle='-')


plt.legend(loc = 'upper right')
plt.xlabel('Country')
plt.ylabel('Effect Value')
plt.title('World Happiness Score and Affecting Factors')
plt.show()
#Scatter Plot
# x = Economy_GDP_per_Capita y = Family
data15.plot(kind='scatter', x='Economy_GDP_per_Capita', y='Family',alpha=0.5,color='r')
plt.xlabel('Economy_GDP_per_Capita')
plt.ylabel('Family')
plt.title('Comparison of Family and Economic Happiness')
plt.show()
#Histogram , bins = number of bar figure
data15.Economy_GDP_per_Capita.plot(kind='hist',bins=40,figsize=(12,12))
plt.show()
data15_Turkey = data15['Country']== 'Turkey' 
data16_Turkey = data16['Country']== 'Turkey'
data17_Turkey = data17['Country']== 'Turkey'

data15[data15_Turkey].head()


data16[data16_Turkey].head()

data17[data17_Turkey].head()