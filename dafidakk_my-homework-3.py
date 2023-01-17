# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataFrame=pd.read_csv("../input/diabetes.csv")
dataFrame.head()
dataFrame.tail()

# run above two line seperatly and see results.

dataFrame.columns
# 'shape' key word gives number of rows and columns in a tuble
dataFrame.shape
# 'info' key word gives data type like dataframe, number of sample or row, number of feature or column, feature types and memory usage
dataFrame.info()
# Lets look frequency of Age feature of dataFrame.
print(dataFrame['Age'].value_counts(dropna =False))  # if there are nan values that also be counted
# As it can be seen below there are 72 person to age 22 have an issue about Diabetes.
# you can also try with "Outcome"..
# For example max Glucose is 199 and max Age is 81
dataFrame.describe() #ignore null entries
#box plot
dataFrame.boxplot(column='Glucose',by = 'Outcome')
plt.show()
#in this box plot
#  upper edge is the  'Glucose' max = 199
#  lower edge is the  'Glucose' min = 0
#  the container upper short edge Q3= %75 upper quantier 'Glucose' = 140
#  the container lower short edge Q1= %25 lower quantier 'Glucose' = 99
#  the container mid plane is median  'Glucose' = 117
#  there are a few outlier by the outcome
# Firstly I create new data from dataFrame data to explain melt more easily.
data_new = dataFrame.head()    # I only take 5 rows into new data
data_new
# melting
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
# column name  variable ve value is default
melted = pd.melt(frame=data_new,id_vars = 'Glucose', value_vars= ['BMI','Age'])
melted
# Reverse of melting.
# Index is Glucose
# after reverse melting i want to next feature name is melted columns variable
# Finally values in columns are value
melted.pivot(index = 'Glucose', columns = 'variable',values='value')
data1 = dataFrame.head()
data2= dataFrame.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row

#ignore index:True giving new indexing number after the concat.

#column concatenating, horizontal 
data1 = dataFrame['BloodPressure'].head()
data2= dataFrame['SkinThickness'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 1 : adds dataframes in column
conc_data_col
dataFrame.dtypes
#  convert float to int, int to float
dataFrame['BMI'] = dataFrame['BMI'].astype('int')
dataFrame['Age']= dataFrame['Age'].astype('float')

dataFrame.dtypes
dataFrame.info()
#cheking some features
dataFrame["Age"].value_counts(dropna=False)
#there was no missing data or NaN value etc.
#other check method is assert
assert dataFrame['Age'].notnull().all() # its rerun nothing because Age feature has no nan values..its like boolean operators. if there is wrong in the statment 
# syntax error came up.
#some other statment with assert
assert dataFrame.Age.dtypes == np.float # same. nothing return.