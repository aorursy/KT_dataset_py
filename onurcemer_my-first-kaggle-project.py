# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as sts # I prefer import Statistic Library for mean method
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015.csv')
data.info() # For learn to our data columns, variable and these types
data.describe()
data.columns 
data.columns = [each.lower() for each in data.columns]
data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]
data.columns
data.rename(columns={'economy_(gdp':'eco_gdp','health_(life':'health_life','trust_(government':'trust_government'},inplace=True)
data.columns
data.corr()
data['health_life'].corr(data['eco_gdp']) #Correlation between Health Life and Gross Domestic Power
data.generosity.plot(kind = 'hist',bins = 30,figsize = (14,14))
plt.show() # We learn "Generosity" frequency
# Line Plot
data.health_life.plot(kind='line' , color ='blue',linewidth=1,alpha=0.5,grid=True,label='HealthLife',linestyle=':')
data.eco_gdp.plot(kind = 'line',color='red',linewidth=1,alpha=0.5,grid=True,label='EconomyPower',linestyle='-')
plt.legend()
plt.xlabel('Rank of Countries')
plt.ylabel('Index of Countries')
plt.show()
# scatter Plot
data.plot(kind='scatter', x='happiness_score', y='eco_gdp',alpha = 0.8,color = 'red')
plt.xlabel('Happiness Score')
plt.ylabel('Economy - Gross Domestic Product')
plt.title('Happines Score - Economy Scatter Plot')          
plt.show()
regions = data.region
print(len(regions.unique())) # I have a different 10 region
print(regions.unique())
easternAsia = data[data.region == 'Eastern Asia']
easternAsia
# We will find the mean of Happiness Score
happinessStatement = sts.mean(easternAsia['happiness_score']) # sts is our statistcis library name
happinessStatement
data['happiness_statement'] = ["happy" if each > happinessStatement else "unhappy" for each in data['happiness_score']]
easternAsia # Eastern Asia has 3 happy 3 unhappy country
easternAsia.generosity.plot(kind='hist',bins=50,figsize=(10,10))
plt.xlabel("Generosity")
plt.title("Histogram")
plt.show() # Generosity frequency in Eastern Asia
gn = sts.mean(easternAsia.generosity) # mean generosity
ep = sts.mean(easternAsia.eco_gdp) # mean ecoPower
print("In Eastern Asia")
print("Mean Economic Power:",ep)
print("Mean Generosity:",gn)
# Economy - Generosity Scatter Plot

data.plot(kind = 'scatter',x='eco_gdp',y='generosity',alpha=0.8,color='green')
plt.xlabel("Economy")
plt.ylabel("Generosity")
plt.title('Economy - Generosity')
plt.show()
easternAsia['eco_gdp'].corr(easternAsia['generosity'])
dictionary = {'Language' : 'English','France' : 'Paris'}
print(dictionary.keys())
print(dictionary.values())

dictionary['Language'] = "Spanish"    # update existing entry
print(dictionary)
dictionary['Turkey'] = "Ankara"       # Add new entry
print(dictionary)
del dictionary['France']              # remove entry with key 'spain'
print(dictionary)
print('Turkey' in dictionary)        # check include or not | True or False
# dictionary.clear()                   # remove all entries in dict
print(dictionary)
q = data['dystopia_residual']>2.5
a = len(data[q])
print (a)
print(data[q])
qw = data[np.logical_and(data['dystopia_residual']>1.5, data['generosity']>0.5 )]
print(qw)
i = 0
while i != 5 :
    print('i is: ',i)
    i +=1 
print(i,' is equal to 5')
for index,value in data[['happiness_score']][10:16].iterrows():
    print(index," : ",value)