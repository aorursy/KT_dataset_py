# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns # 0.9.0 version

import warnings # current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings("ignore")

sns.set(style='darkgrid')

%matplotlib inline
# Load data into a pandas dataframe

iris = pd.read_csv("../input/Iris.csv")



# See first 5 entries of dataframe

iris.head()
# Create a scatter plot with relplot

sns.relplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)
# Create a scatter plot with lmtest

sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)
# Scatter plot without regression line

sns.lmplot(x='SepalLengthCm', y='SepalWidthCm', fit_reg=False, data=iris)
# Create figure

sns.relplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)



# Add title to figure

plt.title('Scatter Plot')



# Change x-axis label

plt.xlabel('Sepal Length (cm)')



# Change y-axis label

plt.ylabel('Sepal Width (cm)')



# Change x-axis range

plt.xlim(5,7)



# Change y-axis range

plt.ylim(2,4);
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', height=7, aspect=1.5, data=iris)



plt.title('Scatter Plot')

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Sepal Width (cm)')



# Condensed way of setting axis limits

plt.axis([5,7,2,4])
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=iris)
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', style='Species', data=iris)
sns.relplot(x='SepalLengthCm', y='SepalWidthCm', col='Species', data=iris)
sns.catplot(x='Species', y='SepalLengthCm', data=iris)
sns.catplot(x='Species', y='SepalLengthCm', jitter=False, data=iris)
sns.catplot(x='Species', y='SepalLengthCm', jitter=False, 

            order=['Iris-virginica', 'Iris-setosa', 'Iris-versicolor'], data=iris)
sns.catplot(x="Species", y="SepalLengthCm", kind="box", data=iris)
sns.catplot(x="Species", y="SepalLengthCm", kind="violin", data=iris)
# First define the violin plot with the insides set to be empty

sns.catplot(x="Species", y="SepalLengthCm", kind="violin", inner=None, data=iris)



# Add scatterplot

sns.swarmplot(x="Species", y="SepalLengthCm", color="k", size=3, data=iris)
# Creating a bar graph using sns.catplot

sns.catplot(x='Species', kind='count', data=iris)
# Creating a bar graph using sns.countplot

sns.countplot(x="Species", data=iris)
# Note that the input is no longer x = "" but just the actual dataframe column

sns.distplot(iris['SepalLengthCm'])
# Create a histogram

sns.distplot(iris['SepalLengthCm'], kde=False)
# Add tick marks

sns.distplot(iris['SepalLengthCm'], kde=False, rug=True)
# Set number of bins equal to 20

sns.distplot(iris['SepalLengthCm'], bins=20)
# Create a kernel density estimator: each observation is replaced with a Gaussian curve centered at its value 

# and the curves are summed to find the density value at the point; the resulting curve is normalized to have an area of 1

sns.distplot(iris['SepalLengthCm'], hist=False, rug=True)
sns.kdeplot(iris['SepalLengthCm'], shade=True)
sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', data=iris)
# Set kind='kde'

sns.jointplot(x='SepalLengthCm', y='SepalWidthCm', kind='kde', data=iris)
# Input is name of dataframe

sns.pairplot(iris, hue='Species')