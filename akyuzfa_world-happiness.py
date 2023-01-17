# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df15 = pd.read_csv('../input/2015.csv')
df16 = pd.read_csv('../input/2016.csv')
df17 = pd.read_csv('../input/2017.csv')
df15["Year"] = '2015'
df16["Year"] = '2016'
df17["Year"] = '2017'
df15.head()
df16.head()
df17.head()
df15.drop(columns = ["Standard Error"], inplace = True)
df16.drop(columns = ["Lower Confidence Interval", "Upper Confidence Interval"], inplace = True)
df17.drop(columns = ["Whisker.high", "Whisker.low"], inplace=True)
df17.rename(columns={'Happiness.Rank': 'Happiness Rank', 'Happiness.Score': 'Happiness Score', 'Economy..GDP.per.Capita.':'Economy (GDP per Capita)', 'Health..Life.Expectancy.':'Health (Life Expectancy)', 'Trust..Government.Corruption.':'Trust (Government Corruption)','Dystopia.Residual':'Dystopia Residual'}, inplace=True)
df17.merge(df16, on='Country', how='right')
df = pd.concat([df15, df16, df17], sort=False)
df
df[df["Happiness Score"]>7.5]
df.set_index('Year')
df.describe()
sum(df.duplicated()) #there is no dublicated item
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f')
x = input('Enter your country:')
df[df['Country'].str.contains(x)]
df.isnull().sum() #i couldnt find a way for region :)
df.Freedom.plot(label = 'Freedom',linewidth=1,alpha = 1,grid = True)