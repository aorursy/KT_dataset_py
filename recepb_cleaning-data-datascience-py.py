import numpy as np
#NumPy is a python library used for working with arrays.
#It also has functions for working in domain of linear algebra, fourier transform, and matrices.
#We have lists that serve the purpose of arrays, but they are slow.NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.

import pandas as pd 
#Why pandas: you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — 
#a table, basically — then let you do things like:
#Calculate statistics and answer questions about the data, like: What's the average, median, max, or min of each column?
#Does column A correlate with column B?
#What does the distribution of data in column C look like?
#Clean the data by doing things like removing missing values and filtering rows or columns by some criteria
#Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.
#Store the cleaned, transformed data back into a CSV, other file or database

import os
#The OS module in python provides functions for interacting with the operating system.
#This module provides a portable way of using operating system dependent functionality.
#The *os* and *os.path* modules include many functions to interact with the file system.

import matplotlib.pyplot as plt
#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

import seaborn as sns

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#UTF-8 is a variable-width character encoding standard 
#that uses between one and four eight-bit bytes to represent all valid Unicode code points.

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.       
data=pd.read_csv("/kaggle/input/fifa19/data.csv")
data.head()
# tail shows last 5 rows
data.tail()
#A quantile defines a particular part of a data set, i.e. a quantile determines how many values in a distribution are above or 
#below a certain limit. Special quantiles are the quartile (quarter), the quintile (fifth) and percentiles (hundredth).
#count: number of entries
#mean: average of entries
#std: standart deviation
#min: minimum entry
##25%: first quantile
#50%: median or second quantile
#75%: third quantile
#max: maximum entry
data.describe()
#Exploratory Data Analysis or (EDA) is understanding the data sets by summarizing their main characteristics often plotting them visually.
#1. Importing the required libraries for EDA(pandas,numpy,seaborn,matplotlib)
#2.Loading the data into the data frame (just read the CSV into a data frame and pandas data frame does the job for us.)
#3. Checking the types of data
data.dtypes
# We have to convert that string to the integer data only then we can plot the data via a graph
#5. Renaming the columns
#drop( self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors="raise" )
#labels: The labels to remove from the DataFrame. It’s used with ‘axis’ to identify rows or column names.
#axis: The possible values are {0 or ‘index’, 1 or ‘columns’}, default 0. It’s used with ‘labels’ to specify rows or columns.
#index: indexes to drop from the DataFrame.

data = data.drop(labels= ['ID', 'Unnamed: 0', 'Photo', 'Flag','GKHandling','GKKicking','SlidingTackle','GKPositioning','GKDiving','Special','Club Logo','StandingTackle','International Reputation'], axis=1)
data.head()
#6. Dropping the duplicate rows
duplicate_rows_data = data[data.duplicated()]
print('number of duplicate rows: ', duplicate_rows_data.shape)
data = data.drop_duplicates()
data.count()
#7. Dropping the missing or null values.
# Dropping the missing values.
data = data.dropna() 
data.count()