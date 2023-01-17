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
plt.style.use("seaborn-whitegrid")
#plt.style.available : To see all the available style in matplotlib library

import seaborn as sns

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#UTF-8 is a variable-width character encoding standard 
#that uses between one and four eight-bit bytes to represent all valid Unicode code points.

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.    
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_PassengerId = test_df["PassengerId"] # Keep original Passenger ID 
train_df.columns
train_df.head()
train_df.info()
def bar_plot(variable):
    """
        input: variable ex: "Sex"
        output: bar plot & value count
    """
    # get feature
    var = train_df[variable]
    # count number of categorical variable(value/sample)
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,varValue))
      
category1 = ["Survived","Sex"]
for c in category1:
    bar_plot(c)  
category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
    print("{} \n".format(train_df[c].value_counts()))
def plot_hist(variable):
    plt.figure(figsize = (9,3))
    plt.hist(train_df[variable], bins = 50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
numericVar = ["Age","PassengerId"]
for n in numericVar:
    plot_hist(n)
# Survived: passenger   Pclass: passenger class
# Sex: gender of passenger  Age: age of passenger
# Fare: amount of money spent on ticket Cabin: cabin category

# Plcass vs Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived",ascending = False)

# Age vs Survived
train_df[["Age","Survived"]].groupby(["Age"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# Sex vs Survived
train_df[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean().sort_values(by="Survived",ascending = False)
# z = (X — μ) / σ 
# Find the mean and standard deviation of the all the data points. 
# And find the z score for each of the data point in the dataset and 
# if the z score is greater than 3 than we can classify that point as an outlier. 
# Any point outside of 3 standard deviations would be an outlier.

dataset= [10,12,12,13,12,11,14,13,
          15,10,10,10,100,12,14,13, 12,10, 10,11,12,15,12,13,12,11,14,13,15,10,15,12,10,14,13,15,10]
outliers=[]
def detect_outlier(data):
    
    threshold=3 
    mean = np.mean(data)            # Mean
    std =np.std(data)               # Standard deviation
    
    for y in data:
        z_score= (y - mean)/std     # Z score = (Observation — Mean)/ Standard Deviation
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

# pass dataset that we created earlier and pass that as an input argument to the detect_outlier function

outlier_datapoints = detect_outlier(dataset)
print(outlier_datapoints)
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile
        Q1 = np.percentile(df[c],25)
        # 3rd quartile
        Q3 = np.percentile(df[c],75)
        # Find the IQR which is the difference between third and first quartile
        IQR = Q3 - Q1
        
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indeces
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
        
    # how many outliers does feature have?
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers

train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
# drop outliers
train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
# FEATURES
# train + test
train_df_len = len(train_df)
train_df = pd.concat([train_df,test_df],axis = 0).reset_index(drop = True)
train_df.head()
# TYPES
train_df.info()
# Standard missing values are missing values that Pandas can detect.
# Going back to our original dataset
# In the row there’re “NAN” values. Clearly these are both missing values. 
train_df.columns
# isnull() method, we can confirm that both the missing value and “NAN” were recognized as missing values. 
# Both boolean responses are True.
# Pandas will recognize both empty cells and “NAN” types as missing values. 
train_df.columns[train_df.isnull().any()]
# Sometimes it might be the case where there’s missing values that have different formats. (n/a NA — naN)
# Pandas will recognize “NAN” as a missing value, but what about the others?
# An easy way to detect these various formats is to put them in a list. Then when we import the data, 
# Pandas will recognize them right away. 
missing_values = ["n/a", "na", "--"]
df = pd.read_csv("/kaggle/input/titanic/train.csv", na_values = missing_values)
print(df['Cabin'])
print("--------------------------------------------")
print(df['Cabin'].isnull())
# Unexpected Missing Values
# if our feature is expected to be a string, but there’s a numeric type, 
# then technically this is also a missing value.
df = pd.read_csv("/kaggle/input/titanic/train.csv")
print(df['Ticket'])
print("--------------------------------------------")
print(df['Ticket'].isnull())
# * Loop through the Ticket column
# * Try and turn the entry into an integer
# * If the entry can be changed into an integer, enter a missing value
# * If the number can’t be an integer, we know it’s a string, so keep going

# Detecting numbers 
cnt=0
for row in df['Ticket']:
    
    try:
        int(row)
        df.loc[cnt, 'Ticket']=np.nan
         # loc method is the preferred Pandas method for modifying entries in place.
            
    except ValueError:
        #If we were to try and change an entry into an integer and it couldn’t be changed,
        #then a ValueError would be returned, and the code would stop. To deal with this,
        #we use exception handling to recognize these errors, and keep going.
        pass
    cnt+=1
    
# If the value can be changed to an integer, we change the entry to a missing value using Numpy’s np.nan.  
# if it can’t be changed to an integer, we pass and keep going.

# After we’ve cleaned the missing values, we will probably want to summarize them. 
# For instance, we might want to look at the total number of missing values for each feature.,
# Total missing values for each feature
print(df.isnull().sum())
# Any missing values?
print(df.isnull().values.any())
# Total number of missing values
print(df.isnull().sum().sum())
train_df[train_df["Embarked"].isnull()]
train_df.boxplot(column="Fare",by = "Embarked")
plt.show()