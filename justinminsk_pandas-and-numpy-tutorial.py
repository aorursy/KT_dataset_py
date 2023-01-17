import pandas as pd

import numpy as np

#import the libraries
iris = pd.read_csv('../input/Iris.csv')

#create a array variable named iris

iris.head()

#display the table
iris.columns

#shows the column titles
iris.describe()

#gives you basic stats of each column
count_speices = pd.value_counts(iris['Species'], sort = True).sort_index()

#counts each unique instance, in this case Iris-setosa, Iris-versicolor, and Iris-virginica 

#can also just use iris['Species'].value_counts(), but the extra code will help with messy code

count_speices

#call the varible to print the count
Sepal = iris['SepalLengthCm'] * iris['SepalWidthCm']

Petal = iris['PetalLengthCm'] * iris['PetalWidthCm']

#create varibles for the area of the petal and sepal, this is not that useful since we are missing height

#however I am using these to show how to concat using panadas
s1 = pd.Series(Sepal, name = 'Sepal')

#adding this extra line to give the column a title 

s2 = pd.Series(Petal, name = 'Petal')

#adding this extra line to give the column a title 

iris_2 = pd.concat([iris, s1, s2], axis=1)

#add our two new varibles to the existing data

iris_2.columns

#double check it added
iris.groupby('Species').mean()

#group by the species and show mean of each column, if you had a second column of strings 

#then it would not display, however if we had two strings you could breakdown the data by both Strings with 

#iris.groupby(['Species', 'String']).mean()

#you can also use many of the other numpy functions like .max(), .min(), .sum(), etc
Iris_setosa = iris_2['Species'].map(lambda x: x.startswith('Iris-setosa'))

Iris_versicolor = iris_2['Species'].map(lambda x: x.startswith('Iris-versicolor'))

Iris_virginica = iris_2['Species'].map(lambda x: x.startswith('Iris-virginica'))

#makes a boolean that finds only one species name, here I create boolean arrays that are true when only 

#one iris species is in the species column for example:

Iris_setosa

#which shows you the array and each true is a row with Iris-setosa, false would be either Iris-versicolor 

#or Iris-virginica
IS = iris_2[Iris_setosa]

IVE = iris_2[Iris_versicolor]

IVI = iris_2[Iris_virginica]

#creates tables for each species for example 

IS

#this shows all 50 entires of Iris-setosa
np.corrcoef(iris_2['PetalLengthCm'], iris_2['PetalWidthCm'])[0, 1]

#Pearson corralation of lengths of petals VS widths of petals in all of our data