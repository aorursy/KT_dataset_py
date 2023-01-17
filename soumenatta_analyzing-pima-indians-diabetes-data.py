# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Any results you write to the current directory are saved as output.
# Import required package

from pandas import read_csv



# Specify the file name 

filename = '/kaggle/input/pima-indians-diabetes-database/diabetes.csv'



# Read the data 

data = read_csv(filename) 



# Print the shape 

data.shape
# Print the first few rows 

data.head()
# Show the type of 'data'

type(data) 
# Get the column names 

col_idx = data.columns

col_idx
# Get row indices 

row_idx = data.index

print(row_idx)
# Find data type for each attribute 

print("Data type of each attribute:")

data.dtypes
# Generate statistical summary 

description = data.describe()

print("Statistical summary of the data:\n")

description
class_counts = data.groupby('Outcome').size() 

print("Class breakdown of the data:\n")

print(class_counts)
# Compute correlation matrix 

correlations = data.corr(method = 'pearson') 

print("Correlations of attributes in the data:\n") 

correlations
skew = data.skew() 

print("Skew of attribute distributions in the data:\n") 

skew
# Import required package 

from matplotlib import pyplot 

pyplot.rcParams['figure.figsize'] = [20, 10] # set the figure size 
# Draw histograms for all attributes 

data.hist()

pyplot.show()
# Density plots for all attributes

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)

pyplot.show() 
# Draw box and whisker plots for all attributes 

data.plot(kind= 'box', subplots=True, layout=(3,3), sharex=False, sharey=False)

pyplot.show()
# Compute the correlation matrix 

correlations = data.corr(method = 'pearson') # Correlations between all pairs of attributes

# Print the datatype 

type(correlations)

# Show the correlation matrix 

correlations
# import required package 

import numpy as np 



# plot correlation matrix

fig = pyplot.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

names = data.columns

ax.set_xticklabels(names,rotation=90) # Rotate x-tick labels by 90 degrees 

ax.set_yticklabels(names)

pyplot.show()
# Import required package 

from pandas.plotting import scatter_matrix

pyplot.rcParams['figure.figsize'] = [20, 20]



# Plotting Scatterplot Matrix

scatter_matrix(data)

pyplot.show()