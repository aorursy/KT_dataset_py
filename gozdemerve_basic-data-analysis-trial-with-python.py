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
data = pd.read_csv("../input/2017.csv")
data.info()
data.dtypes # this also shows us the type of features.
data.head()
#data.corr()

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f', ax=ax)

plt.show()
# we can check all columns name of the table.

data.columns
# Happiness and Economy status are likely correlated. This means we can say that rich people are likely to be happier.



data.plot(kind='scatter', x='Happiness.Score', y='Economy..GDP.per.Capita.',alpha = 0.5,color = 'red')

plt.xlabel('Happiness.Score')              # label = name of label

plt.ylabel('Economy')

plt.title('Happiness and Economy Scatter Plot')       
# Happiness and Family status are likely correlated. This means we can say that family effects people's happiness.



data.plot(kind='scatter', x='Happiness.Score', y='Family',alpha = 0.5,color = 'blue')

plt.xlabel('Happiness.Score')              # label = name of label

plt.ylabel('Family')

plt.title('Happiness and Family Scatter Plot')     
# Happiness and Health status are likely correlated. This means we can say that health effects people's happiness.



data.plot(kind='scatter', x='Happiness.Score', y='Health..Life.Expectancy.',alpha = 0.5,color = 'green')

plt.xlabel('Happiness.Score')              # label = name of label

plt.ylabel('Health..Life.Expectancy.')

plt.title('Happiness and Health Scatter Plot')     
# freedom is also important. Let's check whether freedom also effects happiness.

# according to graph, we can say that they are slightly correlated.



data.plot(kind='scatter', x='Happiness.Score', y='Freedom',alpha = 0.5,color = 'orange')

plt.xlabel('Happiness.Score')              # label = name of label

plt.ylabel('Freedom')

plt.title('Happiness and Freedom Scatter Plot')
data["Happiness.Score"].plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
data.head()
# Ranges of Economy, Health and Family features:



print("Economy:")

print("Average value is: ",data['Economy..GDP.per.Capita.'].mean())

print("Maximum value is: ",data['Economy..GDP.per.Capita.'].max())

print("Minimum value is: ",data['Economy..GDP.per.Capita.'].min())

print("")

print("Health:")

print("Average value is: ",data['Health..Life.Expectancy.'].mean())

print("Maximum value is: ",data['Health..Life.Expectancy.'].max())

print("Minimum value is: ",data['Health..Life.Expectancy.'].min())

print("")

print("Family:")

print("Average value is: ",data["Family"].mean())

print("Maximum value is: ",data['Family'].max())

print("Minimum value is: ",data['Family'].min())
# Rage of Happiness Score

print("Happiness:")

print("Average value is: ",data["Happiness.Score"].mean())

print("Maximum value is: ",data['Happiness.Score'].max())

print("Minimum value is: ",data['Happiness.Score'].min())
# Let's look at the countries which has Economy lower than 0.9 and Happiness score higher than 5.

data[(data['Economy..GDP.per.Capita.']<0.9) & (data['Happiness.Score']>5)]
# Number of country between these values:

print(data[(data['Economy..GDP.per.Capita.']<0.9) & (data['Happiness.Score']>5)]["Happiness.Rank"].count())
# This time, let's look at the countries which has Economy higher than 0.9 and Happiness score lower than 5.

data[(data['Economy..GDP.per.Capita.']>0.9) & (data['Happiness.Score']<5)]
# Number of these countries:

print(data[(data['Economy..GDP.per.Capita.']>0.9) & (data['Happiness.Score']<5)]["Happiness.Rank"].count())
print("Total number of countries:", data["Happiness.Rank"].count())

print("Happy countries with Economics higher than average:",data[(data['Economy..GDP.per.Capita.']>0.9) & (data['Happiness.Score']>5)]["Happiness.Rank"].count())

print("The ratio: ", 83/155)

print()

print("Happy countries with Economics lower than average ",data[(data['Economy..GDP.per.Capita.']<0.9) & (data['Happiness.Score']>5)]["Happiness.Rank"].count())

print("The ratio: ", 15/155)