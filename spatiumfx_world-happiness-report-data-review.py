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



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015.csv')

data.drop(columns={'Happiness Rank'}, inplace=True)

print("DATA FOR 2015")

data.info()
data.corr(method = 'kendall')
data.corr(method ='spearman')  # I used Spearman's correlation coefficient because there were statistically more sensitive and clearer symptoms.
#correlation map

f,ax = plt.subplots(figsize=(24, 10))

sns.heatmap(data.corr(method = "spearman"), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10) # Rank 10
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

from pylab import rcParams

rcParams['figure.figsize'] = 20, 5



data.rename(columns={'Economy (GDP per Capita)': 'Economy'}, inplace=True)

data.rename(columns={'Happiness Score': 'Happiness_Score'}, inplace=True)

data.Economy.plot(kind = "line", color = "red", label = "Economy", linewidth = 2, alpha = 0.9, grid = True, linestyle = ":")

data.Happiness_Score.plot(kind = "line", color = "blue", label = "Happiness Score", linewidth = 2, alpha = 0.9, grid = True, linestyle = ":")

plt.legend(loc = "upper right")

plt.xlabel ("Economy", color = "red")

plt.ylabel ("(Happiness Score Range = [0,8])", color = "blue")

plt.title ("Line Plot")

axes = plt.gca()

axes.set_xlim([0,157])

axes.set_ylim([0,8])

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind = "scatter", x = "Economy", y = "Family", color = "blue", alpha = 1, linewidth = 2, label = "Correlation")

plt.legend(loc ="upper right")

plt.xlabel("Economy", color = "red")

plt.ylabel("Family", color = "green")

plt.title("Economy & Family Scatter Plot")

plt.show()
data.Happiness_Score.describe() # Gives us ideas to read the Histogram
# Histogram

# bins = number of bar in figure

data.Happiness_Score.plot(kind = "hist", bins = 40, figsize = (21,7),color = "purple", alpha = 0.7, label = "Histogram")

plt.legend()

plt.title("Happiness Score Frequency")

plt.xlabel("Happiness Score")

axes = plt.gca()

axes.set_xlim([2.8,7.6])

axes.set_ylim([0,11])

plt.show()
data = pd.read_csv('../input/2015.csv')
series = data['Family']        

print(type(series))

data_frame = data[['Family']]

print(type(data_frame))
# 1 - Filtering Pandas data frame

x = data['Economy (GDP per Capita)']>1.4    # As you can see, the economy is not always happy.

data[x]
data.rename(columns={'Health (Life Expectancy)': 'Health'}, inplace=True)

data.rename(columns={'Happiness Score': 'Happiness_Score'}, inplace=True)

data.Happiness_Score.describe()

data.Health.describe()
# 2 - Filtering pandas with logical_and

# As you can see, the health is not always happy.

data[np.logical_and(data['Health']>0.811013, data['Happiness_Score']< 5.375734 )]