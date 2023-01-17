import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")



#Import libraries

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd



sns.set(style="white", color_codes=True)



# Load the iris data set, which is in the "../input/" directory

iris = pd.read_csv("../input/Iris.csv")



#View the head of the data

iris.head()
# Let's drop the Id column since we don't need it.

iris.drop('Id', axis=1, inplace=True)
# Create emptry grids to plot the data into later

g = sns.PairGrid(iris)

# Then map scatter plots to the grid

g.map(plt.scatter)
# Map to upper,lower, and diagonal

g = sns.PairGrid(iris)

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot)
# Let's create a pairplot with the hue set to 'species' column so we can distinguish the patterns between species

sns.pairplot(iris,hue='Species',palette='Set2')
# Plot petal length and sepal length on a jointgrid

g = sns.JointGrid(x="PetalLengthCm", y="SepalLengthCm", data=iris)

g = g.plot(sns.regplot, sns.distplot)
sns.boxplot(x="Species", y="PetalLengthCm", data=iris, palette='colorblind')
g = sns.FacetGrid(iris,hue="Species",palette='Set2',size=4,aspect=2)

#map to grid:

g = g.map(plt.hist,'PetalLengthCm',bins=15,alpha=0.7)