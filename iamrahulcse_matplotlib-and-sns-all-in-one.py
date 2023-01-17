#%% [markdown]

# # Iris Data Species

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from IPython.display import display

import matplotlib.patches as patch

import matplotlib.pyplot as plt

from sklearn.svm import NuSVR

from scipy.stats import norm

from sklearn import svm

import lightgbm as lgb

import xgboost as xgb

import seaborn as sns

import pandas as pd

import numpy as np

import warnings

import time

import glob

import sys

import os

import gc



#%% [markdown]

# # 

iris=pd.read_csv('../input/Iris.csv')



iris.head()



#%% [markdown]

iris.info()



#%% [markdown]

iris['Species'].value_counts()



#%% [markdown]

# # Creating the bar plot

sns.countplot('Species',data=iris)



#%% [markdown]

# ## Counting the values in the pie plot

iris['Species'].value_counts().plot.pie(figsize=(10,8))



#%% [markdown]

# ## Joint plot: Jointplot is seaborn library specific and can be used to quickly visualize and analyze the relationship between two variables and describe their individual distributions on the same plot.

figure=sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris)



#%% [markdown]

sns.jointplot(x='SepalWidthCm',y='SepalLengthCm',data=iris,kind='reg')



#%% [markdown]

# ## Jointplot's for the Sepal Length and Width

sns.jointplot(x='SepalWidthCm',y='SepalLengthCm',data=iris,kind='hex')



#%% [markdown]

sns.jointplot(x='SepalWidthCm',y='SepalLengthCm',data=iris,kind='resid')



#%% [markdown]

sns.jointplot(x='SepalWidthCm',y='SepalLengthCm',data=iris,kind='kde')



#%% [markdown]

# ## Boxplot for the Species and PetalLengthCm

sns.boxplot(x='Species',y='PetalLengthCm',data=iris)

plt.xlabel('Species of the plant')

plt.title('Box Plot Of Figure')

#%% [markdown]

# ## Strip_plot

sns.stripplot(x='Species',y='PetalLengthCm',data=iris)



#%% [markdown]

# ## Combining both the boxplot and strip_plot

fig=plt.gcf()

fig=sns.boxplot(x='Species',y='SepalLengthCm',data=iris)

fig=sns.stripplot(x='Species',y='SepalLengthCm',data=iris)



#%% [markdown]

# ## Four different kinds of the violin_plots  

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=iris)



#%% [markdown]

# ## Scattterplot

sns.scatterplot(x='Species',y='PetalLengthCm',data=iris)



#%% [markdown]

# ## Pairplot for the iris dataset.

sns.pairplot(data=iris,hue='Species')



#%% [markdown]

# ## Heatmap for the iris dataset.

sns.heatmap(data=iris.corr(),annot=True)



#%% [markdown]

# ## Don't know how to plot the distribution plot??





#%% [markdown]

# ## Swarm Plot

sns.boxplot(x='Species',y='PetalLengthCm',data=iris)

sns.swarmplot(x='Species',y='PetalLengthCm',data=iris)



#%% [markdown]

# ## Lmplot

sns.lmplot(x="PetalLengthCm",y='PetalWidthCm',data=iris)



#%% [markdown]

# # FacetGrid is still incomplete?

sns.FacetGrid(iris,hue='Species')



#%% [markdown]

from pandas.tools.plotting import andrews_curves

andrews_curves(iris,"Species",colormap='rainbow')

plt.ioff()



#%% [markdown]

# ## Parallel coordinate plot: This type of visualisation is used for plotting multivariate, numerical data. Parallel Coordinates Plots are ideal for comparing many variables together and seeing the relationships between them. For example, if you had to compare an array of products with the same attributes (comparing computer or cars specs across different models).

from pandas.tools.plotting import parallel_coordinates



parallel_coordinates(iris,"Species",colormap="rainbow")



#%% [markdown]

# ## Factorplot

sns.factorplot('Species','SepalLengthCm',data=iris)



#%% [markdown]

# ## Boxenplot

sns.boxenplot('Species','SepalLengthCm',data=iris)



#%% [markdown]

fig=sns.residplot('SepalLengthCm', 'SepalWidthCm',data=iris)

#%% [markdown]

# # How to create the venn diagram pls let me know?





#%% [markdown]

# # Spider Graph is still in prgoress?
