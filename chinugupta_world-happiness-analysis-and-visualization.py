# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path1 = "../input/world-happiness/2015.csv"

y_5 = pd.read_csv(path1)
import math 

import scipy.stats as stats

import seaborn as sns

import matplotlib.pyplot as plt
y_5.head()
y_5.info()
y_5.describe()
# Happiness score across Regions



plt.figure(figsize=(12,9))

sns.violinplot(y_5['Happiness Score'], y_5['Region'])

plt.show()
# Getting Countries with more than 84% of Happiness scores

reg_5 = y_5.loc[y_5['Happiness Score'] > 6.3]

reg_5
# Regions with most no. of happy countries

reg_5['Region'].value_counts()
# Getting countries in regions having 6.3+ scores

west_euro = y_5.loc[(y_5['Region'] == 'Western Europe') & (y_5['Happiness Score'] > 6.3)] 

print('Total countries in Western Europe with more than 6.3+ score :',len(west_euro.index))



latin_america = y_5.loc[(y_5['Region'] == 'Latin America and Caribbean') & (y_5['Happiness Score'] > 6.3)]

print('Total countries in Latin America and Caribbean with more than 6.3+ score :',len(latin_america.index))



middle_east = y_5.loc[(y_5['Region'] == 'Middle East and Northern Africa') & (y_5['Happiness Score'] > 6.3)]

print('Total countries in Middle East and Northern Africa with more than 6.3+ score :',len(middle_east.index))



southeast_asia = y_5.loc[(y_5['Region'] == 'Southeastern Asia') & (y_5['Happiness Score'] > 6.3)]

print('Total countries in Southeastern Asia with more than 6.3+ score :',len(southeast_asia.index))
# Creating a dataframe containing regions with countries having 6.3+ scores

top_4 = pd.DataFrame({'Region':['Western Europe', 'Latin America and Caribbean', 'Middle East and Northern Africa', 'Southeastern Asia']

                      ,'Countries':[len(west_euro.index), len(latin_america.index), len(middle_east.index), len(southeast_asia.index)]})



# Visualizing this dataframe

plt.figure(figsize=(10,8))

sns.barplot(x=top_4['Region'],y=top_4['Countries'],data=top_4)

plt.ylabel('Countries with 6.3+ scores')

plt.xlabel('Regions')

plt.xticks(rotation = 75)

plt.title('Regions containing most no. of countries with 6.3+ scores')
#  Getting top 10 happiest countries in the world

happy_10 = y_5.sort_values(by='Happiness Score',ascending=False).head(10)

plt.figure(figsize=(15,10))

sns.barplot(x='Happiness Score',y='Country',data=happy_10,hue='Country')

plt.legend(loc='lower right')

plt.xlabel('Happiness Scores')

plt.ylabel('Countries')

plt.title('Top 10 happiest countries in 2015')

plt.show()
# Getting top 10 unhappy countries in the world

sad_10 = y_5.sort_values(by='Happiness Score',ascending=False).tail(10)

plt.figure(figsize=(15,10))

sns.barplot(x='Happiness Score',y='Country',data=sad_10,hue='Country')

plt.legend(loc='lower right')

plt.xlabel('Happiness Scores')

plt.ylabel('Countries')

plt.title('Top 10 unhappy countries in 2015')

plt.show()
#  Getting top 10 happiest countries' Economy in the world

happy_10 = y_5.sort_values(by='Happiness Score',ascending=False).head(10)

plt.figure(figsize=(15,10))

sns.barplot(x='Economy (GDP per Capita)',y='Country',data=happy_10,hue='Country')

plt.legend(loc='lower right')

plt.xlabel('Economy (GDP per Capita)')

plt.ylabel('Countries')

plt.title('Top 10 happiest countries with their Economy in 2015')

plt.show()
# Getting top 10 unhappy countries' Economy in the world

sad_10 = y_5.sort_values(by='Happiness Score',ascending=False).tail(10)

plt.figure(figsize=(15,10))

sns.barplot(x='Economy (GDP per Capita)',y='Country',data=sad_10,hue='Country')

plt.legend(loc='lower right')

plt.xlabel('Economy (GDP per Capita)')

plt.ylabel('Countries')

plt.title('Top 10 unhappy countries with their Economy in 2015')

plt.show()
#  Getting top 10 happiest countries' Generosity in the world

happy_10 = y_5.sort_values(by='Happiness Score',ascending=False).head(10)

plt.figure(figsize=(15,10))

sns.barplot(x='Generosity',y='Country',data=happy_10,hue='Country')

plt.legend(loc='lower right')

plt.xlabel('Generosity')

plt.ylabel('Countries')

plt.title('Top 10 happiest countries with their Generosity in 2015')

plt.show()
#  Getting top 10 unhappy countries' Generosity in the world

sad_10 = y_5.sort_values(by='Happiness Score',ascending=False).tail(10)

plt.figure(figsize=(15,10))

sns.barplot(x='Generosity',y='Country',data=sad_10,hue='Country')

plt.legend(loc='lower right')

plt.xlabel('Generosity')

plt.ylabel('Countries')

plt.title('Top 10 unhappy countries with their Generosity in 2015')

plt.show()
# Checking correlation between different variables

plt.figure(figsize=(15,12))

sns.heatmap(y_5.corr(), cmap = 'copper', annot = True)

plt.show()
plt.figure(figsize=(15,12))

west_europe = y_5.loc[lambda y_5 : y_5['Region'] == 'Western Europe']

sns.heatmap(west_europe.corr(), cmap = 'Greys', annot = True)

plt.show()
sns.jointplot('Family', 'Freedom', data=west_europe, kind='kde', space=0)
sns.jointplot('Trust (Government Corruption)', 'Economy (GDP per Capita)', data=west_europe, kind='kde', space=0, color='g')
sns.jointplot('Trust (Government Corruption)', 'Freedom', data=west_europe, kind='kde', space=0)
plt.figure(figsize=(15,12))

east_asia = y_5.loc[lambda y_5 : y_5['Region'] == 'Eastern Asia']

sns.heatmap(east_asia.corr(), cmap = 'pink', annot = True)

plt.show()
sns.jointplot('Health (Life Expectancy)', 'Economy (GDP per Capita)', data=east_asia, kind='kde', space=0, color='g')
plt.figure(figsize=(15,12))

middle_east = y_5.loc[lambda y_5 : y_5['Region'] == 'Middle East and Northern Africa']

sns.heatmap(middle_east.corr(), cmap = 'Blues', annot = True)

plt.show()
plt.figure(figsize=(15,12))

north_america = y_5.loc[lambda y_5 : y_5['Region'] == 'North America']

sns.heatmap(north_america.corr(), cmap = 'rainbow', annot = True)

plt.show()
plt.figure(figsize=(15,12))

africa = y_5.loc[lambda y_5 : y_5['Region'] == 'Sub-Saharan Africa']

sns.heatmap(africa.corr(), cmap = 'Wistia', annot = True)

plt.show()
# Getting top 10 countries in each sector

cols = ['Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)', 'Generosity']



for col in cols:

    print(y_5[['Country', col]].sort_values(by = col,

    ascending = False).head(10))

    print("\n")