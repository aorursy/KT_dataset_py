# Importing the libraries



import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
# Importing the dataset



iris = pd.read_csv("../input/iris/Iris.csv")
# First 3 rows



iris.head(3)
# Last 3 rows



iris.tail(3)
# Size of the dataset



print('(Rows, Columns)=',iris.shape)
# Basic information about data and its datatypes



iris.info()
# Descibing the data by displaying the count, mean, standard deviation & percentile stats



iris.describe()
# fetching column names in the dataset



iris.columns
# Checking the no of null values in the dataset



iris.isnull().sum()
# Fetching columns having null values



iris.columns[iris.isnull().any()]
# Drop the unwanted columns.

# In dataset we have ID columns which represents the record number which can be removed



iris.drop(labels='Id', axis=1, inplace=True)
# Display the first 3 rows of dataset



iris.head(3)
# Checking the different categories and count of species



iris.Species.value_counts()
# Min values of different attributes of iris group by species



iris.groupby('Species').min()
# Cross verify above table with the below value for getting min value of Iris-sentosa species SepalLenthCm

iris[iris['Species']=='Iris-setosa']['SepalLengthCm'].min()
# Max values of different attributes of iris group by species



iris.groupby('Species').max()
# Mean values of different attributes of iris group by species



iris.groupby('Species').mean()
# Plotting graph with mean values according to the species



fig= plt.figure(figsize=(10,5))

for i in iris.columns[:-1]:

    plt.plot(iris.groupby('Species').mean()[i])

plt.legend()
# Distribution of SepalLength of Iris-setosa



sns.distplot(iris[iris['Species']=='Iris-setosa']['SepalLengthCm'], kde=False, bins=20)
# Distribution of PetalLengthCm of Iris-setosa



sns.distplot(iris[iris['Species']=='Iris-setosa']['PetalLengthCm'], kde=False, bins=15)
# Distribution of PetalWidthCm of Iris-setosa



print(sns.distplot(iris[iris['Species']=='Iris-setosa']['PetalWidthCm'], kde=False, bins=15))
# Pair plot displaying each and every attribute according to the species



sns.pairplot(iris, hue='Species')
# Scatter Plot (Sepal length Vs Sepal width)



fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor', ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_title("Sepal length VS Sepal width")

fig.set_xlabel("sepal length")

fig.set_ylabel("sepal width")

fig=plt.gcf()

fig.set_size_inches(15,8)
# Scatter plot (Petal length VS Petal width)



fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor', ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_title("Petal Length VS Petal Width")

fig.set_xlabel("petal length")

fig.set_ylabel("petal width")

fig=plt.gcf()

fig.set_size_inches(15,8)