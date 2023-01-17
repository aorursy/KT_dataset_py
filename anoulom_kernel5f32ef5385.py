# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df2017 = pd.read_csv('../input/world-happiness/2017.csv')

df2017
df2016 = pd.read_csv('../input/world-happiness/2016.csv')

df2016
df2015 = pd.read_csv('../input/world-happiness/2015.csv')

df2015
#importer les librairies nécessaires

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
colonnes2017 = list(df2017.columns.values)

print (colonnes2017)

print(df2017.shape)
#selection des colonnes qui m'interessent pour en faire des corrélations

chth2017 = df2017[['Country', 'Happiness.Score', 'Trust..Government.Corruption.', 'Health..Life.Expectancy.']]

chth2017.head()
colonnes2016 = list(df2016.columns.values)

print (colonnes2016)

print(df2016.shape)
#selection des colonnes qui m'interessent pour en faire des corrélations

crhth2016 = df2016[['Country', 'Region', 'Happiness Score', 'Trust (Government Corruption)', 'Health (Life Expectancy)']]

crhth2016.head()
colonnes2015 = list(df2015.columns.values)

print (colonnes2015)

print(df2015.shape)
#selection des colonnes qui m'interessent pour en faire des corrélations

crhth2015 = df2015[['Country', 'Region', 'Happiness Score', 'Trust (Government Corruption)', 'Health (Life Expectancy)']]

crhth2015.head()
#merge des années 2017, 2016 et 2015



#df2017 = pd.read_csv('../input/world-happiness/2017.csv')

#df2016 = pd.read_csv('../input/world-happiness/2016.csv')

#df2015 = pd.read_csv('../input/world-happiness/2015.csv')



df2017 = df2017[['Country', 'Happiness.Rank', 'Happiness.Score', 'Economy..GDP.per.Capita.', 'Family',

       'Health..Life.Expectancy.', 'Freedom', 

       'Trust..Government.Corruption.', 'Generosity', 'Dystopia.Residual']]

df2017.columns = df2017.columns.str.replace(".", " ")

df2016 = df2016[['Country', 'Happiness Rank', 'Happiness Score',

       'Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom',

       'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']]

df2016 = df2016.reindex(sorted(df2016.columns), axis=1)

df2015 = df2015[['Country', 'Happiness Rank', 'Happiness Score',

       'Economy (GDP per Capita)', 'Family',

       'Health (Life Expectancy)', 'Freedom',

       'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']]

df2015 = df2015.reindex(sorted(df2015.columns), axis=1)

df2017 = df2017.rename(index=str, columns={"Economy  GDP per Capita ": "Economy (GDP per Capita)", "Health  Life Expectancy ":"Health (Life Expectancy)", "Trust  Government Corruption ": "Trust (Government Corruption)"})

df2017 = df2017.reindex(sorted(df2017.columns), axis=1)





df5 = df2017.append([df2016, df2015])

df5
#import du csv suicide

df = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")

df
#merge suicide avec df5 (df5 = merge de 2017, 2016 et 2015)

df5 = pd.merge(df, df2017, left_on='country', right_on='Country', how='left')



df5.dropna(axis=1, how='all')

df5.drop(columns=['HDI for year', 'Country', 'country-year'], axis=1, inplace=True)

df5
#class4 = df5.groupby('year').country.value_counts()

#class4
#sns.catplot(x='Trust (Government Corruption)', y=crhth2016['Region'], data=df5 , kind="bar")
sns.catplot(x='Trust (Government Corruption)', y="Region", data=crhth2016 , kind="bar")
sns.catplot(x='Health (Life Expectancy)', y="Region", data=crhth2016 , kind="bar")
#sns.catplot(x='Trust (Government Corruption)', y='Health (Life Expectancy)', data=crhth2016, height=6, kind="strip")



#(x="class", y="survived", hue="sex", data=titanic, height=6, kind="bar", palette="muted")
sns.lmplot(x='Trust (Government Corruption)', y='Health (Life Expectancy)', data=crhth2016, height=6)
