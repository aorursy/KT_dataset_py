#This code was retrieved from https://www.kaggle.com/benhamner/python-data-visualizations. 

#I have inserted some of my own comments to further explain what is going on and made some minor changes to some of the charts and graphs. This is meant to be for practice and was a homework assignment for a Data Visualization class. 

#The code utilizes the pandas and seaborn packages for the visualizations.

##to show the visualizations in the jupyter notebook

%matplotlib inline
# First, import pandas, a data analysis tool for visualizations

import pandas as pd



# We'll also import seaborn, a Python statistical data visualization library

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt



# Next, we'll load the Iris flower dataset, which is in the same directory as this file.

iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame



# Now we will take a look at the iris DataFrame we have created to se if we imported it correctly.

iris.head()
# We will now see how many measurements we have for each species to get a better idea of our dataset

iris["Species"].value_counts()
# The first way we can plot things is using the .plot extension from Pandas dataframes

# Use this to make a scatterplot of the Iris features - Sepal Length and Sepal Width

iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
# We can also use the seaborn library to make a similar plot

# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure

# we want the x axis to be Sepal Length and the y ais to be Sepal Width, using the iris dataset we imported

# we will also convert the kind of graph to "kde" which makes the plot a heat map and change the color to green.

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, kind="kde", color="g")
# One piece of information missing in the plots above is what species each plant is

# We'll use seaborn's FacetGrid to color the scatterplot by species

sns.FacetGrid(iris, hue="Species", size = 6).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend() #adds a legend so you can see which color corresponds to which species
# I'm going to recreate the plot from above but by using Petal Width and Length instead of Sepal Width and Length

sns.FacetGrid(iris, hue="Species", size = 6).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend() #adds a legend so you can see which color corresponds to which species
# We can look at an individual feature in Seaborn through a boxplot

# We will look at Petal length between the species to get an idea of how the lengths are distributed

sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
# One way we can extend this plot is adding a layer of individual points on top of

# it through Seaborn's striplot

# 

# By using jitter=True we can view all the points without those points being on the same axis.

# This is simply an aesthetic method to help increase the effectiveness of the visualization



# Saving the resulting axes as ax each time causes the plot to be shown on top of the previous axes

ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)

ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")
# A violin plot combines the benefits of the previous two plots and simplifies them

# Denser regions of the data are fatter, and sparser thiner in a violin plot.

#This further illustrates the distribution of Petal lengths between species

sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
# A final seaborn plot useful for looking at univariate relations is the kdeplot,

# which creates and visualizes a kernel density estimate of the underlying feature

#I added some shading for some aesthetics

sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm", shade=True).add_legend() #adds the legend for the species
# Another useful seaborn plot is the pairplot, which shows the bivariate relation

# between each pair of features

# 

# From the pairplot, we'll see that the Iris-setosa species is separataed from the other

# two across all feature combinations

# we'll drop the Id column from thr iris dataset because that column is not an index, not a measurement 

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)
# The diagonal elements in a pairplot show the histogram by default

# We can update these elements to show other things, such as a kde which may be a little more clear.

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")
# I will recreate the above plot with some shading on the diagonal kde plots, I will also change the points to "+" signs

sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde",diag_kws=dict(shade=True), markers = "+")
# We can quickly make a boxplot with Pandas on each feature split out by species

# this visualization is great because it allows you to get a good idea of the ditribution of all the measurements quickly

# figsize helps control the height and width of the output plot 

iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))