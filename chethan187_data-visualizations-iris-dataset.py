## This notebook gives basic data visualization techniques using popular libraries Seaborn and Matplotlib

## Source https://seaborn.pydata.org/examples/index.html and https://matplotlib.org/tutorials/index.html

## Data Source: Popular Iris dataset 



import pandas as pd 

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
## Import and read the data 

df = pd.read_csv('../input/Iris.csv') 

df.head()
# Pairplot for the entire datasets



sns.pairplot(df) #Gives the visual repreesntations of the complete data
# Heatmap to understand the data correlations



sns.heatmap(df.corr(), annot=True, linewidths=.5)
## Counter library for the counts for each Iris type



from collections import Counter

Species_count = Counter(df['Species'])

Species_count



dict(Species_count)
# Barplot for each Iris type



plt.bar(Species_count.keys(), Species_count.values(), color='g') #To know the species count
## Jointplot for two variables.Can be used to understand the linear realtionship of two variables lin linear regression analysis



sns.jointplot(data=df, kind='scatter', x='SepalLengthCm', y='SepalWidthCm')
## Boxplot for displaying data summary in five number summary 

sns.set_style('whitegrid')



sns.boxplot(data=df, x='Species', y='PetalLengthCm')
## Boxplot with one dimentional scatter plots using 'Jitter' 



sns.set_style('whitegrid')



sns.boxplot(data=df, x='Species', y='PetalLengthCm')

sns.stripplot(data=df, x='Species', y='PetalLengthCm', jitter=True)
plt.style.use('seaborn-whitegrid')



sns.relplot(x='PetalLengthCm', y='PetalWidthCm', data=df, sizes=(40,400), alpha=.5, palette='muted',

            height=6)
## Scatter plot using hue=Species 



plt.style.use('seaborn-whitegrid')



g = sns.scatterplot(x='PetalLengthCm', y='PetalWidthCm', data=df, hue='Species', sizes=(10,200), legend='brief')



box = g.get_position()

g.legend(loc='upper left')
## Multiple bivariate KDE Plots 



sns.set(style='darkgrid')



setosa = df.query("Species == 'Iris-setosa'")

virginica = df.query("Species == 'Iris-virginica'")



f, ax = plt.subplots(figsize=(8,8))

ax.set_aspect('equal')



sns.kdeplot(setosa.SepalLengthCm, setosa.SepalWidthCm, cmap='Reds', shade=True, shade_lowest=False)

sns.kdeplot(virginica.SepalLengthCm, virginica.SepalWidthCm, cmap='Blues', shade=True, shade_lowest=False)



ax.text(4.5, 4.2, "setosa", color='Red')

ax.text(7.5, 2.3, "virginica", color='blue')
## Matplotlib horizontal plot



plt.rcParams.update({'figure.autolayout': True})



fig, ax = plt.subplots()

ax.barh(df['Species'], df['PetalLengthCm'], align='center')

labels = ax.get_xticklabels()
## Comparing Sepal and Petal's length and width



plt.figure()



fig, ax = plt.subplots(1,2, figsize=(15, 5))



df.plot(x='SepalLengthCm', y='SepalWidthCm', kind='scatter', ax=ax[0], sharex=False, color='red')

df.plot(x='PetalLengthCm', y='PetalWidthCm', kind='scatter', ax=ax[1], sharex=False, color='blue')



ax[0].set(title='Sepal Comparison')

ax[1].set(title='Petal Comparison')