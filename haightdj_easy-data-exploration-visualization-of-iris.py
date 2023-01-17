# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

%matplotlib inline
sns.set_style("whitegrid")

# Supress warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
DF_iris = pd.read_csv("../input/Iris.csv") # Load Iris data into pandas DataFrame
print(DF_iris.keys()) # Print a list of the index keys
print(DF_iris.shape) # Return the number of (rows, columns) in the dataset
DF_iris.head() # Shows the first 5 rows of data
DF_iris.describe() # Show some very basic stats on the numerical data columns
# Check for any missing data:
print('Missing Training Data:')
DF_iris.isnull().sum()
# First, Let's drop the ID column from the DF, since that is not going to be interesting to explore
DF_iris.drop(labels = ['Id'], axis = 1, inplace = True)
DF_iris.head()
vars = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
print(vars)
print(DF_iris.Species.value_counts())
sns.countplot(DF_iris.Species);
# We can use built-in features of a Pandas DataFrame to plot the distributions of all of the variables...
# However, these are not the prettiest plots...
DF_iris.hist(grid=False)
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()
grid = sns.FacetGrid(pd.melt(DF_iris[vars]), col = 'variable', col_wrap = 4, size = 5.0, 
                    aspect = 1.0, sharex = False, sharey = False)
grid.map(sns.distplot, 'value')
plt.show;
plt.figure(figsize=(25,5))
plt.subplot(1,4,1)
sns.boxplot(x='Species',y='PetalLengthCm',data=DF_iris)
plt.subplot(1,4,2)
sns.boxplot(x='Species',y='PetalWidthCm',data=DF_iris)
plt.subplot(1,4,3)
sns.boxplot(x='Species',y='SepalLengthCm',data=DF_iris)
plt.subplot(1,4,4)
sns.boxplot(x='Species',y='SepalWidthCm',data=DF_iris);
sns.boxplot(x='Species',y='SepalWidthCm',data=DF_iris)
sns.stripplot(x="Species", y="SepalWidthCm", data=DF_iris, jitter=True, edgecolor="gray"); # jitter shifts point slightly left-right so there isn't too much overlap
plt.figure(figsize=(25,5))
plt.subplot(1,4,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=DF_iris)
plt.subplot(1,4,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=DF_iris)
plt.subplot(1,4,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=DF_iris)
plt.subplot(1,4,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=DF_iris);
# Let's plot 2 variables against eachother...
g = sns.jointplot("PetalLengthCm", "PetalWidthCm", data=DF_iris, kind="reg")
g = sns.lmplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=DF_iris, fit_reg=False)
g = sns.pairplot(DF_iris, hue='Species', height=4)
# g.map_upper(sns.regplot) # some plot options: 'regplot', 'residplot', 'scatterplot'
#g.map_lower(sns.kdeplot)
#g.map_diag(plt.hist)

from pandas.plotting import parallel_coordinates

plt.figure(figsize=(15,10))
parallel_coordinates(DF_iris[['Species', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']],'Species')