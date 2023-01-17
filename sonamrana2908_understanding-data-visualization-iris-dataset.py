# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Loading the Data

iris = pd.read_csv('../input/Iris.csv')
#Checking the head of Dataset

iris.head(5)
#Checking the column info

iris.info()
#to check the statisics for the iris dataset

iris.describe()
iris.groupby('Species').min()
iris.groupby('Species').max()
#Checking is there is any null data in iris dataset

iris.isnull().any()

# There is no missing Data.
iris.groupby('Species').count()
#using countplot to see number of each kind of iris species flower

sns.countplot(x=iris['Species'])
iris['PetalLengthCm'].hist(bins=20)
iris.drop('Id',axis=1).hist(bins=20,figsize=(10,10))
iris.groupby('Species').PetalLengthCm.plot.hist(alpha=0.4)

plt.xlabel('PetalLengthCm')

plt.suptitle('Histogram of PetalLengthCm for different Species')

plt.legend(loc=(0.69,0.75))

plt.grid()
iris.groupby('Species').PetalWidthCm.plot.hist(alpha=0.4)

plt.xlabel('PetalWidthCm')

plt.suptitle('Histogram of PetalWidthCm for different Species')

plt.legend(loc=(0.69,0.75))

plt.grid()
#Histogram using Matplotlib,Pandas and Seaborn.

plt.hist(data=iris,x='PetalLengthCm',bins=20)#Matplotlib

iris['PetalLengthCm'].hist(bins=20)# Pandas

sns.distplot(iris['PetalLengthCm'],bins=20) #Seaborn
 #to see the relationship between 2 variables.

sns.set_style('darkgrid')

sns.scatterplot(data=iris,x='PetalLengthCm',y='PetalWidthCm',hue='Species')
sns.scatterplot(data=iris,x='PetalLengthCm',y='SepalLengthCm',hue='Species')
#Since this data is not that big and has only 4 features, we can use pairplot 

#and check the relationshipo between 2 variables grouped by Species column.

sns.pairplot(data=iris.drop('Id',axis=1),hue='Species')
#Scatter Plot using Matplotlib,Pandas and Seaborn.

#sns.scatterplot(data=iris,x='PetalLengthCm',y='SepalLengthCm',hue='Species')#Seaborn.Hue can be used only here.

#plt.scatter(data=iris,x='PetalLengthCm',y='PetalWidthCm',c='green',marker='+')#matplotlib

#iris.plot.scatter(x='SepalLengthCm',y='SepalWidthCm',c='red',marker='+',s=10)#pandas

#iris.groupby('Species').plot.scatter(x='SepalLengthCm',y='SepalWidthCm',c='red',marker='+',s=10)#pandas

#but this will result in 3 separate plots. Hence using seaborn hue is easier and more visually appealing

# as it will plot the data on single axis.

iris.drop('Id',axis=1).corr()
#Plotting the correlation using HeatMap

sns.heatmap(iris.drop('Id',axis=1).corr(),cmap='viridis',annot=True,)

plt.suptitle('Heatmap')
sns.boxplot(data=iris,y='SepalLengthCm')
sns.boxplot(data=iris,x='Species',y='SepalLengthCm')
sns.violinplot(data=iris,y='SepalWidthCm')
sns.violinplot(data=iris,x='Species',y='SepalLengthCm')
#Subplot

fig,axes = plt.subplots(ncols=2,nrows=1)# Creating the grid

axes[0].hist(iris['PetalLengthCm'])# Plotting on each axis

axes[0].set_title('PetalLengthCm')# Setting the Title

axes[1].scatter(iris['PetalLengthCm'],iris['PetalWidthCm'])# Plotting on each axis

axes[1].set_title('ScatterPlot')# Setting the Title
#Facet

g=sns.FacetGrid(data=iris,col='Species')# this creates the blank grid based on level of categorical variable.

g.map(plt.hist,'PetalLengthCm')# plotting using the grid created.

# grid can also be created using 2 Categorical Variables and span over rows.
h=sns.FacetGrid(data=iris,col='Species')

h.map(plt.scatter,'PetalLengthCm','PetalWidthCm',color='r')# for bivariate(2 variable) plotting