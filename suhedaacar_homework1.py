# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Reading "World Happiness Report" 2017 csv file

data = pd.read_csv('../input/2017.csv')
data.info()

data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
# Line Plot
data["Freedom"].plot(kind = 'line', color = 'g',label = 'Freedom',linewidth=1,alpha = 0.5,grid = True)
data["Dystopia.Residual"].plot(color = 'r',label = 'Dystopia.Residual',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('Line Plot')
plt.show()
# Scatter Plot 
# x = Family, y = Happiness Score
data.plot(kind='scatter', x='Family', y='Happiness.Score',alpha = 0.5,color = 'red')
plt.xlabel('Family')  
plt.ylabel('Happiness.Score')
plt.title('Happiness Score - Family Scatter Plot')
# Histogram
data["Happiness.Score"].plot(kind = 'hist',bins = 50,figsize = (12,8))
plt.show()
data["Happiness.Score"].plot(kind = 'hist',bins = 50)
plt.clf()
# Happiness score mean value
h_s_mean = data['Happiness.Score'].mean()
print(h_s_mean)
# Create a new column as happiness range
data["Happiness.range"] = ["below_average" if h_s_mean > each else "above_average" for each in data['Happiness.Score']]
data.head()
# Seperate below average and above average happiness score
fltr = data["Happiness.range"] == "below_average"
data_below_average = data[fltr]
print(data_below_average)

fltr2 = data["Happiness.range"] == "above_average"
data_above_average = data[fltr2]
print(data_above_average)


maxValue = data["Happiness.Score"].max()
print(maxValue)

for index,value in data[['Country']][0:3].iterrows():
    print(index," : ",value)
