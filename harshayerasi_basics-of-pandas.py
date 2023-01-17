# Importing pandas into the notebook.
import pandas as pd
import numpy as np
# Pandas series
sampleSeries = pd.Series([1,2,3,4,5])
print('The series is')
print(sampleSeries, ' and the type of the series is',type(sampleSeries),'and the elements are of type',sampleSeries.dtype)
# You can also convert a python list to pandas series
list1 = [1,2,3,4,5,6,7]
series1 = pd.Series(list1)
print(series1, ' and the type of the series is',type(series1),'and the elements are of type',series1.dtype)
# Accessing the elements of a series
print(series1[0])
# Index of pandas series starts from 0
# you can also give names to the indexes
series1.index=['index0','index1','index2','index3','index4','index5','index6']
# You can access the elements of the series using those names.
series1.index0
# You can also access the elements this way
series1['index0']
# You can also access multiple index elements at the same time.
series1[['index1', 'index4', 'index0']]
# You can also access elements from a range of index
series1['index1': 'index5']
# You can get the elements of a series by applying conditons on them.
# Here consider we want all the elements of the series that are grater than 3.
series1[series1>3]
# Here consider we want all the elements of the series that are less than 4.
series1[series1<4]
# You can also perform logical operations.
series1[(series1>3) & (series1>4)] # Round brackets are mandatory when you perform logical operations
series1[(series1>3) | (series1>4)]
# You you want to perform an arithmentic operation on all the elements you can do this.
series1.add(1)
# FInd the maximum value element's index by using below command.
series1.idxmax()
# Find the minimum value element's index by using below command.
series1.idxmin()
# Try to get the value of the maximum by using above index.

# Prints the first 5 values
series1.head()
# you can select how many values to print by passing the parameter
series1.head(2)
# Prints the last 5 values
series1.tail()
# you can select how many values to print by passing the parameter
series1.tail(2)
# A brief statistical analysis on the data
series1.describe()
# I have already imported Students Performance in exams dataset into the kernel.
# Use the below command to read the data file into the notebook.
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
# Printing the first 5 rows of the data..
df.head()
# Printing last 5 rows of the data
df.tail()
# If there are no headers or if you want to ignore headers use the below command.
dfWithoutHeader = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv',header=None)
dfWithoutHeader.head()
# Read csv assumes that the dataset is seperated by commas, if the dataset is separated by ';'' you can use the below command.
# dfSeperatedBySemicolon = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv',';')
# If the data is seperated with whitespace you can use the below command
dfWithWhitespace = pd.read_csv('../input/uci-autompgdataoriginal/auto-mpg.data-original',delim_whitespace=True,header=None)
dfWithWhitespace.head()
# Tells you the number of rows and columns
df.shape
df.shape[0] # Number of instances or rows in the dataset
df.shape[1] # Number of columns or attributes in the dataset
df.columns
# Prints all the data
# Use loc to get the data from the dataframe with reference to column names
df.loc[:,:]
# Accessing the first element of the data frame
df.loc[0,['writing score']]
# Accessing all the rows of a column
df.loc[:,['writing score']]
# You can also slice the rows
df.loc[2:5,:]
# similar to loc, iloc is used to get data from the dataframe with reference to the index of the columns
df.iloc[0:2,[0,1,2]]
# Accessing particular rows only
df.iloc[[1,10,15],[0,1,2]]
# We can also perform filtering on the data
onlyMale = df['race/ethnicity'][df.gender == 'male']
onlyMale.head()
# y includes our labels/Target/Independent variables  and X includes our features/Dependent variables
y = df['writing score']                 
non_inputs = ['race/ethnicity','writing score']
X = df.drop(non_inputs,axis = 1 )
X.head()
# GIves the statistical nalysis of a dataframe, provided the columns must not be of type object
df.describe()
# You can also do statistical analysis based on the columns
df['math score'].describe()
# Provides the information about the dataset
df.info()
correlation = df.corr()
correlation
# Importing packages that will provide us data visualization.
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print(sns.heatmap(correlation))
sns.pairplot(df)