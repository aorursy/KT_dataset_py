import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/ramen-ratings/ramen-ratings.csv")
df.head()
df.shape
df.info()
df.columns
df.describe(include='all')
df['Stars'].describe()
#Since the 'Stars' is in object data type instead of int, Convert it into int

df['Stars'] = pd.to_numeric(df['Stars'],errors='coerce')

df.describe(include='all')
df['Stars'].describe()
#check if there are any NULL values in the data

df.isnull().sum()
#replace the null values with '0'

df.fillna(0,inplace=True)
df.isnull().sum()
df['Top Ten'].head()
#Drop the Columns that are not usefull

df.drop('Top Ten',axis=1,inplace=True)
df.columns
#convert brand to lower case

df['Brand'] = df['Brand'].str.lower()

df.head()
df.Country
df.Country.unique()
#number of countries 

print(len(df.Country.unique()))
#Different styles on ramen 

df['Style'].unique()
df['Style'].value_counts()
#number of ramen brands given

print(len(df.Brand.unique()))
#top ten brands of ramen noodles

df['Brand'].value_counts()[:10]
#lets select the style of ramen noodles with stars above 4

style = df.Style[df.Stars > 4]
#the different styles of noodles with Stars above 4

style.value_counts()
# Ramen noodles in Japan with Stars above 4

j = df.loc[(df.Country=='Japan') & (df.Stars>4)]

j
sns.set(style='whitegrid')

f,ax = plt.subplots(1,1,figsize=(20,5))

sns.countplot(x='Brand',data=j)

plt.xticks(rotation=90)

plt.show()