# import the required libraries



import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# path to the data



filepath="../input/irish-dataset/irish.csv"
# read the data into a variable called irish_data

irish_data=pd.read_csv(filepath)
# Print the first 5 rows of the data

irish_data.head()


# Set the width and height of the figure

plt.figure(figsize=(16,6))

# line plot for all the numerical variables

sns.lineplot(data=irish_data.select_dtypes(exclude='object'))

# Add title

plt.title("line plot for all the numerical variables")
# Line plot a subset of the data

# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Line plot showing 'sepal length'

sns.lineplot(data=irish_data['sepal length'],label='sepal length')

# Line plot showing 'sepal width'

sns.lineplot(data=irish_data['sepal width'],label='sepal width')

# Add title

plt.title("line plot of sepal length and sepal width")



# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Bar plot showing sepal length of the flower type 'Iris-setosa'

sns.barplot(x=irish_data.index[irish_data[' flower type']=='Iris-setosa'], y=irish_data['sepal length'][irish_data[' flower type']=='Iris-setosa'])

# Add title

plt.title("Bar plot showing sepal length of the flower type 'Iris-setosa'")


# correlation between sepal length, sepal width, petal length and petal width

irish_data_corr=irish_data.select_dtypes(exclude='object').corr()

# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Heatmap showing the correlation between sepal length, sepal width, petal length and petal width. Annot=True ensures that the values for each cell appear on the heatmap

sns.heatmap(data=irish_data_corr,annot=True)

plt.title("heatmap of correlation matrix between sepal length, sepal width, petal length and petal width")


# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Ccatter plot between sepal length and sepal width. The scatterplot suggests that sepal length and sepal width are negatively correlated. Correlation coeefficient of -0.12 in heatmap supports this 

sns.scatterplot(x=irish_data['sepal length'],y=irish_data['sepal width'])

# Drawing a regression line confirms the relationship

sns.regplot(x=irish_data['sepal length'],y=irish_data['sepal width'])

# Scatter plot

# Set the width and height of the figure

plt.figure(figsize=(14,6))

# scatter plot between sepal length and petal length. The scatterplot suggests that sepal length and petal length are positively correlated. Correlation coeefficient of 0.87 in heatmap supports this 

sns.scatterplot(x=irish_data['sepal length'],y=irish_data[' petal length '])

# Drawing a regression line confirms the relationship

sns.regplot(x=irish_data['sepal length'],y=irish_data[' petal length '])
# Scatter plots can display the relationships between more than two variables through parameter 'hue'.

# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Scatter plot between sepal length and petal length. In addition to showing positive correlation between sepal length and petal length, 

# the scatterplot suggests that the flowers having higher petal length have also higher petal width 

# which is also supported by the correlation coefficient of 0.96 in heatmap between petal length and petal width

sns.scatterplot(x=irish_data['sepal length'],y=irish_data[' petal length '],hue=irish_data['petal width'])

# Swarmplot can be considered as the scatter plot to feature a categorical variable

# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Swarmplot showing the petal length for different flower types. The plot suggests that 'Iris-virginica' has the highest petal lenth on average

sns.swarmplot(x=irish_data[' flower type'],y=irish_data[' petal length '])


# Set the width and height of the figure

plt.figure(figsize=(14,6))

# histogram to see how sepal width varies in iris flowers

sns.distplot(irish_data['sepal width'],kde=False)

# title

plt.title("histogram of sepal width")
# different histograms each for one flower type

sns.distplot(irish_data[' petal length '][irish_data[' flower type']=='Iris-setosa'],label='Iris-setosa',kde=False)

sns.distplot(irish_data[' petal length '][irish_data[' flower type']=='Iris-versicolor'],label='Iris-versicolor',kde=False)

sns.distplot(irish_data[' petal length '][irish_data[' flower type']=='Iris-virginica'],label='Iris-virginica',kde=False)

plt.legend()

# It suggests that 'Iris-virginica' has the highest petal lenth on average as seen also in the swarm plot
# kernel density estimate (KDE) shows the pdf of a variable. It can be considered as a smoothed histogram as well

# Set the width and height of the figure

plt.figure(figsize=(14,6))

# KDE plot of sepal width. shade=True colors the area below the curve

sns.kdeplot(irish_data['sepal width'],shade=True)

# Title

plt.title("Probability distribution function of sepal width")
# KDE plot for each species of the flowers

plt.figure(figsize=(14,6))

sns.kdeplot(irish_data[' petal length '][irish_data[' flower type']=='Iris-setosa'],label='Iris-setosa')

sns.kdeplot(irish_data[' petal length '][irish_data[' flower type']=='Iris-versicolor'],label='Iris-versicolor')

sns.kdeplot(irish_data[' petal length '][irish_data[' flower type']=='Iris-virginica'],label='Iris-virginica')

# The legend does not automatically appear on the plot. It is forced by the following command

plt.legend()

# Add title

plt.title("PDF of petal lengths by flower types")

# The plot suggests that Iris versicolor and Iris virginica have an intersection of their petal length values,

# while Iris setosa has totally different petal length distribution indicating that petal length might be used to differentiate/classify Iris setosa from other two categories
# KDE plots can be created simultaneously for two variables with the sns.jointplot commandr

# The color-coding shows how likely it is to see different combinations of sepal width and sepal length with darker parts of the figure being the more likely.

# Set the width and height of the figure

plt.figure(figsize=(14,6))

# Jointplot for sepal lenght and sepal width

sns.jointplot(x=irish_data["sepal length"],y=irish_data['sepal width'],kind='kde')
