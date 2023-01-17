import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('../input/cardataset/data.csv')

#top five rows

df.head(5)
#bottom 5 rows

df.tail(5)
df.dtypes
df = df.drop(['Engine Fuel Type', 'Market Category', 'Vehicle Style', 'Popularity', 'Number of Doors','Vehicle Size'], axis=1)

df.head(5)
df = df.rename(columns={"Engine HP": "HP", "Engine Cylinders": "Cylinders", "Transmission Type": "Transmission", "Driven_Wheels": "Drive Mode","highway MPG": "MPG-H", "city mpg": "MPG-C", "MSRP": "Price" })

df.head(5)
df.shape
duplicate_rows_df = df[df.duplicated()]

print("number of duplicate rows: ", duplicate_rows_df.shape)
# dropping duplicate rows

df = df.drop_duplicates()

df.head(5)
print(df.isnull().sum())
df = df.dropna()              # dropping the values

df.count()
print(df.isnull().sum())       # after dropping the values
sns.boxplot(x=df['Price'])
sns.boxplot(x=df['HP'])
sns.boxplot(x=df['Cylinders'])
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
df.Make.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))

plt.title("Number of cars by make")

plt.ylabel('Number of cars')

plt.xlabel('Make');

fig, ax = plt.subplots(figsize=(10,6))

ax.scatter(df['HP'], df['Price'])

ax.set_xlabel('HP')

ax.set_ylabel('Price')

plt.show()
plt.figure(figsize=(10,5))

c = df.corr()

sns.heatmap(c, cmap="BrBG", annot=True )