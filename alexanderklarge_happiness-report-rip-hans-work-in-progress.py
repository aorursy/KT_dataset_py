import numpy as np  

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

#will import plotly at a later date, recreate their bubble charts and "choropleth" plots



import os

print(os.listdir("../input"))

df_2015 = pd.read_csv('../input/2015.csv')

df_2016 = pd.read_csv('../input/2016.csv')

df_2017 = pd.read_csv('../input/2017.csv')
df_2015.head(4)
df_2015.describe()
#Plot distribution of 'Freedom' across the dataset

 

sns.set_style('darkgrid')

sns.set_palette('muted')

plt.figure(figsize=(12,6))

sns.distplot(df_2015['Freedom'],bins=25)
#Plot distribution of 'Health' across the dataset



plt.figure(figsize=(12,6))

sns.distplot(df_2015['Health (Life Expectancy)'],bins=15)
#plot 'Happiness Score' vs 'Economy'

sns.jointplot(x='Happiness Score',y='Economy (GDP per Capita)',data=df_2015)
x = df_2015['Economy (GDP per Capita)'].idxmax()

df_2015.iloc[x]

#I'm sure there's a way to do this in one line of code!
#get a linear regression plot

sns.jointplot(x='Happiness Score',y='Economy (GDP per Capita)',data=df_2015,kind='reg')
plt.figure(figsize=[10,10])

chart = sns.barplot(x='Region',y='Happiness Score',data=df_2015)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45) #handy bit of code I found that makes x labels legible!
plt.figure(figsize=[12,12])

chart = sns.boxplot(x='Region',y='Happiness Score',data=df_2015)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45) #handy bit of code I found that makes x labels legible!
df_2015[df_2015['Happiness Score']<3]
#best economy of 2015

#df_2015['Economy (GDP per Capita)'].max() #this was my first attempt, but just returns the number. 

#Could then search for that,but bulky 



df_2015.loc[df_2015['Economy (GDP per Capita)'].idxmax()]
# Seperate dataframes for generous and not-generous (less than average)

df_2015['Generosity'].mean()

df_2015_gen = df_2015[df_2015['Generosity']>df_2015['Generosity'].mean()] #it worked! Hallelujah!

df_2015_notgen = df_2015[df_2015['Generosity']<df_2015['Generosity'].mean()] 
df_2015_gen.head()
df_2015_notgen.head()