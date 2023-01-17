# https://github.com/tanmoyie/Applied-Statistics

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# ploting related libraries
import matplotlib.pyplot as plt
import plotly.plotly as plty
import plotly.graph_objs as go
#modeling
from sklearn import datasets, linear_model
# load the EXCEL file & read the data 
dataframe1 = pd.read_csv("../input/Take home exam 1 201736034.csv")
print(dataframe1)
print(dataframe1.dtypes)
input_data = dataframe1.values
input_data_asMatrix = dataframe1.as_matrix()


# Examine the properties of the dataset
# print the statistics
print(dataframe1['Total mark (300)'].describe())

# Replacing NaN values
input_adding_roll = dataframe1.iloc[3,10]
print(input_adding_roll)
dataframe1.iloc[3,10] = 15
print(dataframe1.values)
#roll : 201736034
# Scatter plot of CT-3 vs Total mark in CT
print("-------------Scatter Plot: 'CT-3' vs Total mark in CT -----------")
variable_x1 = input_data[:,3] # column 4 ( CT-3 )
variable_y1 = input_data[:,5] # column 6 ( Class Test Total of best 3, Marks: 60 ) 
plt.scatter(variable_x1, variable_y1) # Scatter plot for CT-3 vs Total mark in CT
plt.show() 
# Scatter plot of 'Roll' vs 'Mark total 300'
print("------------- Scatter Plot:'Roll' vs 'Mark total 300' -----------")

variable_x2 = input_data[:, 1]  # column 2  ( CT-1 Mark )
variable_y2 = input_data[:, 11] # column 12 (Mark total 300)
plt.scatter(variable_x2, variable_y2) # Scatter plot of 'Roll' vs 'Mark total 300'
plt.show()

# basic plot
dataframe1.boxplot()
# https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.hist.html
dataframe1.hist()