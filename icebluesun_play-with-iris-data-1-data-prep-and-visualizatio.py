# Import Pandas and set warning settings off



import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook



import pandas as pd # feed data into a dataframe

import numpy as np # calculations and sorting



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Import data and drop unwanted fields



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

print(" ")

print("= First Look ==")

iris=pd.read_csv('../input/Iris.csv')

print(iris.info())



print("===============")

print(" ")



print("= Drop ID =====")

iris.drop('Id',axis=1,inplace=True)

print(iris.info())



print("===============")

print(" ")



# Some data prep



target = ['Species']

remove = ['Id']

xvar  = [i for i in iris.columns if i not in target and i not in remove]



print("Features")

print(xvar)

print("Target")

print(target)



# Look at first 10 rows

iris.head(n=10)
print("General Statistics of Iris Dataset")

print(iris.describe())

print(" ")



for i in xvar:

    print(" Statistics for "+i)

    print(iris[['Species',i]].groupby('Species').describe())

    print(" ")
# Histogram



iris.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
# Box Plots



iris.boxplot(by="Species", figsize=(20, 10))

plt.show()
# Sepal Length VS Width Scatterplot by Species



fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
# Petal Length VS Width Scatterplot by Species



fig = iris[iris.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

iris[iris.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

iris[iris.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
# Violin plots of features by species



plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)

plt.show()