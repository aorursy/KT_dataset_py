import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.pyplot import cm

#import Librarys
iris = pd.read_csv('../input/Iris.csv')

#import data
color_dict = {'Iris-setosa':'blue', 'Iris-virginica':'red', 'Iris-versicolor':'black'}

#create a color dictionary

iris['Species'].value_counts() 

#count of species in the data
x = np.corrcoef(iris['PetalWidthCm'], iris['SepalWidthCm'])[0, 1]

#Pearson corralation of width of petals and sepal in all the data

iris.plot(kind = "scatter", x = "SepalWidthCm", y = "PetalWidthCm", c = [color_dict[i] for i in iris['Species']])

plt.legend([color_dict])

#Comparing width of sepal and petal 

#blue is setosa. red versicolor, and black virginica

print('corralation of width of petals and sepals in all the data:',x)
x = np.corrcoef(iris['SepalLengthCm'], iris['SepalWidthCm'])[0, 1]

#Pearson corralation of area of sepal and sepal+Petal in all the data

iris.plot(kind = "scatter", x = "SepalLengthCm", y = "SepalWidthCm", c = [color_dict[i] for i in iris['Species']])

plt.legend([color_dict])

#plot Sepal length by Sepal width

#blue is setosa. red versicolor, and black virginica

print('corralation of length and width of sepals in all the data:',x)
x = np.corrcoef(iris['PetalLengthCm'], iris['SepalLengthCm'])[0, 1]

#Pearson corralation of lengths of petals and sepal in all the data

iris.plot(kind = "scatter", x = "SepalLengthCm", y = "PetalLengthCm", c = [color_dict[i] for i in iris['Species']])

plt.legend([color_dict])

#plot of lengths of petal and sepal 

print('corralation of lengths of petals and sepals in all the data:',x)
x = np.corrcoef(iris['PetalLengthCm'], iris['PetalWidthCm'])[0, 1]

#Pearson corralation of petal length and width in all the data

iris.plot(kind = "scatter", x = "PetalLengthCm", y = "PetalWidthCm", c = [color_dict[i] for i in iris['Species']])

plt.legend([color_dict])

#plot Petal length by Petal width

print('corralation of length and width of the petals in all the data:',x)
Iris_setosa = iris['Species'].map(lambda x: x.startswith('Iris-setosa'))

Iris_versicolor = iris['Species'].map(lambda x: x.startswith('Iris-versicolor'))

Iris_virginica = iris['Species'].map(lambda x: x.startswith('Iris-virginica'))

#makes a boolean that finds only one species name

IS = iris[Iris_setosa]

IVE = iris[Iris_versicolor]

IVI = iris[Iris_virginica]

#creates tables for each species
heatmap, xedges, yedges = np.histogram2d(IS['PetalLengthCm'], IS['PetalWidthCm'])

#set up a heatmap of Iris setosa petal length by width

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]



plt.clf()

plt.title('Heat Map of Petal Length and Width of Iris Setosa')

plt.xlabel('Petal Length in CM')

plt.ylabel('Petal Width in CM')

plt.imshow(heatmap, extent=extent, cmap=cm.viridis)

plt.colorbar(cmap=cm.viridis)

plt.show()

#the graph shows that the most common width and length is around 0.4 CM width and 1.2 CM Length



heatmap, xedges, yedges = np.histogram2d(IVE['PetalLengthCm'], IVE['PetalWidthCm'])

#set up a heatmap of Iris versicolor petal length by width

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]



plt.clf()

plt.title('Heat Map of Petal Length and Width of Iris Versicolor')

plt.xlabel('Petal Length in CM')

plt.ylabel('Petal Width in CM')

plt.imshow(heatmap, extent=extent, cmap=cm.viridis)

plt.colorbar(cmap=cm.viridis)

plt.show()

#the graph shows the most common width and length is around 1.2 CM width and 4.4 length CM



heatmap, xedges, yedges = np.histogram2d(IVI['PetalLengthCm'], IVI['PetalWidthCm'])

#set up a heatmap of Iris virginicar petal length by width

extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]



plt.clf()

plt.title('Heat Map of Petal Length and Width of Iris Virginica')

plt.xlabel('Petal Length in CM')

plt.ylabel('Petal Width in CM')

plt.imshow(heatmap, extent=extent, cmap=cm.viridis)

plt.colorbar(cmap=cm.viridis)

plt.show()

#the graph shows the most common width and length is around 2.4 CM width and 5.5 CM length