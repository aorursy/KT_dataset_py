#importing necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#now we will load the FIFA 19 dataset

data=pd.read_csv("../input/data.csv")
#lets see the summary of the dataset

data.describe()
#first three rows of the dataset

data.head(3)
#last three rows

data.tail(3)
#lets see how many rows and columns we have in our dataset

print("Number of (rows,columns):",data.shape)
#checking if there is any NULL value in the dataset

data.isna().sum()
#we saaw that most of the data in 'Loaned From' column is not assigned, hence we will drop it

data.drop('Loaned From',axis=1,inplace=True)
#now the data which have NA values, we will fill them with the mean value of that column

data.fillna(data.mean(),inplace=True)
#we will check again if after assigning the mean value to the cells of the originally NA values; if there is any cell which has NA value

data.isna().sum()
#there are still cells in which the mean value could not be assigned. This may be because those columns have strings. So we will assign a value "Unassigned" to the dataset

data.fillna("Unassigned",inplace=True)
#after assigning the term, we shall check again whether we have attained a clean data set or not

data.isna().sum()
#as we started our analysis with the summary of the dataset. We will make a heatmap for the same.

plt.figure(figsize=(50,50))

p=sns.heatmap(data.corr(),annot=True)
# Lets see the top 15 country-wise distribution of players

fif_countries = data['Nationality'].value_counts().head(15).index.values

fif_countries_data = data.loc[data['Nationality'].isin(fif_countries),:]
#we will make a simple visualization for the 15 countries data

#We will make a basic Bar Plot

sns.set(style="dark")

plt.figure(figsize=(25,10))

p=sns.barplot(x='Nationality',y='Overall',data=fif_countries_data)

p.set(xlabel='Country', ylabel='Total')
#Box Plot

sns.set(style="ticks")

plt.figure(figsize=(25,10))

p=sns.boxplot(x='Nationality',y='Overall',data=fif_countries_data)

p.set(xlabel='Country', ylabel='Total')
ten_countries = data['Nationality'].value_counts().head(10).index.values

ten_countries_data = data.loc[data['Nationality'].isin(ten_countries),:]

sns.set(style="ticks")

plt.figure(figsize=(15,10))

p=sns.boxplot(x='Nationality',y='Potential',data=ten_countries_data)