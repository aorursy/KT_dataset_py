# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/vgsales.csv')

df.head(20)
df.shape
df.info()
df.isnull().count()
plt.figure(figsize=(10,10))

plt.xticks(rotation=45)

sns.countplot(df.Genre)

#So We find out that Action Games were made the most 
plt.figure(figsize=(15,10))

plt.xticks(rotation=45)

sns.countplot(df.Year)

#From Year 2000 most number of games were made maybe due to advancement in Technology
#Lets find out for which Platform games were developed the most! 

plt.figure(figsize=(15,10))

plt.xticks(rotation=45)

sns.countplot(df.Platform)
#Let us Analyse which genre games were produced the most in these years 

df.Genre.value_counts()
#Now we will find out which publisher has produced most number of games

df['Publisher'].value_counts()

#So we find out that Electronic Arts is Publisher who had produced the most number of games
#As we now know that Electronic Arts is the biggest game producing Company

#We can use that to analyse which type of games their customers prefer. 

#I mean which genre EA makes most of its games it will be the area of our interest

ea = df[df.Publisher =='Electronic Arts']

ea
ea_genre = ea['Genre'].value_counts()

ea_genre

#So according to my analysis EA produce most games in Sports Genre which means customers of their company

#Loved their Sports games
#Visualizing our results for better Understanding 

plt.figure(figsize=(15,8))

plt.scatter(ea_genre.index,ea_genre.values,s=125,marker='o')

plt.xlabel('Genre')

plt.ylabel('Total')
#Now we will Analyse another famous game Publisher which is Ubisoft

us = df[df.Publisher =='Ubisoft']

us
#So we find out that Ubisoft speciality is in Action genre

us_genre = us['Genre'].value_counts()

us_genre
#Visualizing our results for better Understanding 

plt.figure(figsize=(15,8))

plt.scatter(us_genre.index,us_genre.values,s=125,marker='o')

plt.xlabel('Genre')

plt.ylabel('Total')
#Now we will make a heat map to observe Global Sales of each publisher

relation = df.groupby('Publisher').Global_Sales.sum()

excel_table = pd.pivot_table(df[df.Publisher.isin(relation.sort_values(ascending=False)[:20].index)],values=['Global_Sales'],index=['Year'],columns=['Publisher'],aggfunc='sum',margins=False)

#************************************************************************************

excel_table
plt.figure(figsize=(19,15))

sns.heatmap(excel_table['Global_Sales'],linewidths=.5,annot=True,vmin=0.01,cmap='coolwarm')

#By observing this heatmap we can find Global sales of each Publisher of each year