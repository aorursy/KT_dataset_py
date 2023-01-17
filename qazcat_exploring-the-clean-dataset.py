import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Cleaning data according to my previous kernel

df = pd.read_csv('../input/bgg_db_2017_04.csv', encoding='latin1')

df = df[(df['min_players'] >= 1) & (df['max_players'] >= 1) & (df['avg_time'] >= 1) & (df['min_time'] >= 1) & (df['max_time'] >= 1)]

df = df[df['max_players'] < 99]
df.head()
df.info()
df.describe()
print("Most owned games:")

print(df[['names','owned','rank']].sort_values(by='owned',ascending=False).head(10))
import matplotlib.pyplot as plt

import seaborn as sns



sns.heatmap(df.corr()[['owned']].drop('owned'),cmap='coolwarm',annot=True)

plt.title('Correlation with Number Owned')
sns.heatmap(df.corr()[['geek_rating']].drop('geek_rating'),cmap='coolwarm',annot=True)

plt.title('Correlation with Geek Rating')
def top_in_cat(dataframe,series,number):#Find top game which includes label within categorical data

    data = dataframe[series].values[0].lstrip().split(',')

    for cat in data:

            print('\nTop games with ' + series + ': ' + cat)

            print(df[df[series].apply(lambda x: cat in x)]['names'].head(number))

    

top_in_cat(df,'mechanic',5)#Top games with same Mechanics as Top ranked game
top_in_cat(df,'category',5)#Top games with same category as Top ranked game
top_in_cat(df,'designer',5)#Top games with same designer as Top ranked game