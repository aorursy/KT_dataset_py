# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data15 = pd.read_csv('../input/world-happiness/2015.csv')

data16 = pd.read_csv('../input/world-happiness/2016.csv')

data17 = pd.read_csv('../input/world-happiness/2017.csv')
data15.info()

print()

data16.info()

print()

data17.info()

print()
display(data15.corr())

display(data16.corr())

display(data17.corr())
#correlation map

f,ax = plt.subplots(figsize=(21,15))

sns.heatmap(data15.corr(), annot= True, linewidths=.01, fmt='.1f', ax=ax)

plt.title('2015')

plt.show()



f,ax = plt.subplots(figsize=(21,15))

sns.heatmap(data16.corr(), annot= True, linewidths=.01,fmt='.1f',ax=ax)

plt.title('2016')

plt.show()



f,ax= plt.subplots(figsize=(21,15))

sns.heatmap(data17.corr(), annot=True, linewidths=.01, fmt='.1f', ax=ax)

plt.title('2017')

plt.show()

display(data15.head(10))

display(data16.head(10))

display(data17.head(10))

display(data15.columns)

display(data16.columns)

display(data17.columns)
#Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data15['Economy (GDP per Capita)'].plot(kind = "line", color = "red",label = "Economy (GDP per Capita)",linewidth = 1, alpha = 0.6, grid = True,figsize=(12,6), linestyle = ":")

data15['Health (Life Expectancy)'].plot( color = "blue",label = "Health (Life Expectancy)",linewidth = 1, alpha = 0.5, grid = True,figsize=(12,6), linestyle = "-.")

plt.legend(loc='upper right')

plt.xlabel('Economy')

plt.ylabel('Health (Life Expectancy)')

plt.title('Line Plot 2015')

plt.show()



data16['Economy (GDP per Capita)'].plot(kind = "line", color = "red",label = "Economy (GDP per Capita)",linewidth = 1, alpha = 0.6, grid = True,figsize=(12,6), linestyle = ":")

data16['Health (Life Expectancy)'].plot( color = "blue",label = "Health (Life Expectancy)",linewidth = 1, alpha = 0.5, grid = True,figsize=(12,6), linestyle = "-.")

plt.legend(loc='upper right')

plt.xlabel('Economy')

plt.ylabel('Health (Life Expectancy)')

plt.title('Line Plot 2016')

plt.show()



data17['Economy..GDP.per.Capita.'].plot(kind = "line", color = "red",label = "Economy..GDP.per.Capita.",linewidth = 1, alpha = 0.6, grid = True,figsize=(12,6), linestyle = ":")

data17['Health..Life.Expectancy.'].plot( color = "blue",label = "Health..Life.Expectancy.",linewidth = 1, alpha = 0.5, grid = True,figsize=(12,6), linestyle = "-.")

plt.legend(loc='upper right')

plt.xlabel('Economy')

plt.ylabel('Health (Life Expectancy)')

plt.title('Line Plot 2017')

plt.show()
#Scatter Plot

# x = freedom, y = happiness score

data15.plot(kind = 'scatter', x='Freedom', y='Happiness Score', alpha=0.5,figsize = (12,6), color = 'red')

plt.xlabel('Freedom ')           # label = name of label

plt.ylabel('Happiness Score')

plt.title('Freedom Happiness Score Scatter Plot -2015')   # title = title of plot



data16.plot(kind='scatter', x='Freedom', y='Happiness Score',alpha = 0.5,figsize = (12,6), color = 'blue')

plt.xlabel('Freedom')

plt.ylabel('Happiness Score')

plt.title('Freedom Happines Score Scatter Plot -2016')



data17.plot(kind='scatter', x = 'Freedom', y = 'Happiness.Score', alpha='0.5',figsize = (12,6), color='green')

plt.xlabel('Freedom')

plt.ylabel('Happiness Score')

plt.title('Freedom Happiness Score Scatter Plot -2017')

plt.show()

#Histogram

# bins = number of bar in figure

data15.rename(columns={"Economy (GDP per Capita)" : "Economy"}, inplace=True)

data15.Economy.plot(kind = 'hist', bins = 50, figsize = (12,12),color = 'red')

plt.title('2015')

plt.show() 



data16.rename(columns={"Economy (GDP per Capita)" : "Economy"}, inplace=True)

data16.Economy.plot(kind = 'hist', bins = 50, figsize = (12,12), color='blue')

plt.title('2016')

plt.show()



data17.rename(columns={"Economy..GDP.per.Capita." : "Economy"}, inplace = True)

data17.Economy.plot(kind = 'hist',bins = 50, figsize = (12,12), color = 'green')

plt.title('2017')

plt.show()

# 1 - Filtering Pandas data frame

x = data15['Generosity']>0.5

data15[x]



# 2 - Filtering pandas. We can use '&' for filtering.

data17.rename(columns={"Economy..GDP.per.Capita." : "Economy"}, inplace = True)

data17[ (data17['Happiness.Score']>4) & (data17['Economy']>1.4 )]