# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Data Read

data = pd.read_csv('../input/world-happiness/2017.csv')
#General information about data obtained

data.info()
#Data Correlation

data.corr()
#Data Correlation Heat Map

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt='.2f',ax=ax)

plt.show()
#A pice of data 

data.head(10)
#Information of numeric features

data.describe()
#Features

data.columns
freedom = data['Freedom']

trust = data['Trust..Government.Corruption.']
#Line Plot

#Health..Life.Expectancy. and Economy..GDP.per.Capita.

freedom.plot(kind='line',color='r',label='Freedom',linewidth=1,alpha=0.8,grid=True,linestyle=':')

trust.plot(kind='line',color='g',label='Trust..Government.Corruption.',linewidth=1,alpha=0.8,grid=True,linestyle='-.')

plt.legend()

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.show()
health = data['Health..Life.Expectancy.']

economy = data['Economy..GDP.per.Capita.']
#Scatter Plot 

#Health..Life.Expectancy. - Economy..GDP.per.Capita.

data.plot(kind='scatter',x='Health..Life.Expectancy.' , y='Economy..GDP.per.Capita.' , alpha=0.5 , color='blue')

plt.xlabel('Health..Life.Expectancy.')

plt.ylabel('Economy..GDP.per.Capita.')

plt.title('Health..Life.Expectancy. and Economy..GDP.per.Capita. Scatter Plot ')
#Histogram

happinessScore = data['Happiness.Score']

happinessScore.plot(kind='hist',bins=50 , figsize=(12,12))

plt.show()
happinessScore = data['Happiness.Score'] 

x = happinessScore > 7   

data[x]
freedom = data['Freedom']

y = freedom < 0.1

data[y]
z = (freedom < 0.45) & (happinessScore > 7) 

data[z]
family = data[['Family']]

for index,value in family[0:2].iterrows():

    print(index,":",value)

threshold = sum(data.Family)/len(data.Family)

print("threshold: ",threshold)

data["Family_level"] = ["high" if i>threshold else "low" for i in data.Family]

data.loc[70:80,["Family_level","Family"]]