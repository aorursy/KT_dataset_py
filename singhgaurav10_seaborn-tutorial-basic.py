# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
iris = sns.load_dataset('iris')
titanic = sns.load_dataset('titanic')
# preview of the iris dataset
iris.head()
# lets try to visualize features sepal length & width to better understand the data
sns.relplot(x= 'sepal_length', y= 'sepal_width', data=iris)
# let us add another dimension - the species dimension using 'hue'
sns.relplot(x= 'sepal_length', y='sepal_width', hue='species', data=iris)
# visualizing petal length & width
sns.relplot(x= 'petal_length', y='petal_width', size='species', data=iris)
# creating a time series dataset
time = pd.DataFrame(dict(time = np.arange(500), value = np.random.randn(500).cumsum()))
# plotting the time series using a line chart
sns.relplot(x= 'time', y= 'value', kind= 'line', data= time)
# Preview of the titanic dataset
titanic.head()
# when there are multiple measurements for the same value 
# ex. in the Titanic database- mulitple 'fare' values for the same 'class'
# seaorn creates a confidence interval to represent them
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', data = titanic)
# The CI (confidence interval) feature can be turned off using: None
# It can be changed to standard deviation to represent the distribution at each classification using: 'sd'

sns.relplot(x= 'pclass', y= 'fare', kind= 'line', ci= None, data = titanic)
# segregating the fare v. pclass according to sex of the passenger
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', hue='sex', data= titanic)
# we can add another variable in this if required using 'event' property
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', hue='sex', style= 'survived', ci= None, data= titanic)
# Changing the color palette
# For a comprehensive list of color palettes, kindly visit the documentation. Link provided at the beginning
palette = sns.cubehelix_palette(light = .7, n_colors= 2)
sns.relplot(x= 'pclass', y= 'fare', kind= 'line', hue='sex', style= 'survived', palette= palette, ci= None, data= titanic)
# We use the 'col' feature to create a multi plot graph
# Here each graph/column represents a different classification and the 'x' and 'y' are plotted for each 
sns.relplot(x= 'age', y= 'fare', col='survived', data= titanic)
# we can add additional features, as we did earlier
sns.relplot(x= 'age', y= 'fare', col='survived', hue= 'sex', data= titanic)
# Default plot with catplot is scatterplot
# This helps in visualizing categorical variables
sns.catplot(x= 'sex', y= 'age', data= titanic)
# We can control the magnitude of jitter using the 'jitter' feature
sns.catplot(x= 'sex', y= 'age', jitter= False, data= titanic)
# For small datasets, we can check the distribution of the data using 'swarm' plot
sns.catplot(x= 'sex', y= 'age', kind= 'swarm', data= titanic)
# we can further add more features in the plot using options like 'hue'
sns.catplot(x= 'survived', y= 'age', kind= 'swarm', hue= 'sex', data= titanic)
# visualizing a subset of the data
sns.catplot(x= 'survived', y= 'age', kind= 'swarm', hue= 'sex', data= titanic.query('pclass==1'))
# Age wise distribution in each class
sns.catplot(x= 'pclass', y= 'age', kind= 'box', data= titanic)
# Adding additional feature 
sns.catplot(x= 'pclass', y= 'age', hue= 'sex', kind= 'box', data= titanic)
# A better distribution plot is boxen plot for larger datasets, it sshows the shape of distribution as well
sns.catplot(x= 'pclass', y= 'age', kind= 'boxen', data= titanic)
# A violin plot provides distribution along with IQR plot embedded in it
sns.catplot(x= 'pclass', y= 'age', kind= 'violin', data= titanic)
# We can combine swarm and violin plot to show individual points in the distribution
g = sns.catplot(x= 'pclass', y= 'age', kind= 'violin', inner= None, data= titanic)
sns.swarmplot(x= 'pclass', y= 'age', color= 'k', data= titanic, ax= g.ax)
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)
# When we want to show the number of observations in each category without creating a quantitative variable
sns.catplot(x= 'deck', kind= 'count', palette= 'ch:.25', data= titanic)
# Adding more variables in our analysis
sns.catplot(x= 'class', kind= 'count', hue='sex', palette= 'pastel', data= titanic)
sns.catplot(x= 'sex', y= 'survived', hue= 'class', kind= 'point', data= titanic)
sns.catplot(x= 'class', y= 'age', hue='sex', col= 'survived', kind= 'swarm', data= titanic)
# Creating a variable with gaussian distribution
x= np.random.normal(size= 100)
# Plotting a histogram and a kernel density estimate
sns.distplot(x)
# Plotting a histogram along with a small vertical tick at each observation
sns.distplot(x, bins=20, kde= False, rug= True)
# In multi dimensional data, a useful approach is to draw multiple instances of the same plot on different subsets of the dataset
# The followng command is used to initialize the FacetGrid object with dataframe and row, column, hue 
g= sns.FacetGrid(titanic, col= 'survived', hue= 'sex')
# Adding the kind of plot and features to visualize using the 'map' function
g.map(plt.scatter, 'deck', 'age')
g.add_legend()
# We can change features like height and aspect to alter the look and feel of the plot
g= sns.FacetGrid(titanic, col= 'survived', height= 5, aspect= .6)
g.map(sns.barplot, 'sex', 'fare')
g= sns.PairGrid(iris)
g.map(plt.scatter)
# We can add other features and properties to enhance the visualization
g= sns.PairGrid(iris, hue= 'species', height= 3, aspect= 0.7)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()