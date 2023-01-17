# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import plotly.plotly as plty
import plotly.graph_objs as go
from sklearn import datasets, linear_model
dataframe1 = pd.read_csv("../input/US_graduate_schools_admission_parameters_Dataset1 34.csv")
print(dataframe1)
print(dataframe1.dtypes)
input_data = dataframe1.values
input_data_asMatrix = dataframe1.as_matrix()
print(dataframe1['GRE Score'].describe())
#print(dataframe1['TOEFL Score'].describe())
#print(dataframe1['Chance of Admit'].describe())
print("-------------Scatter Plot: 'GRE Score' vs 'Chance of Admit'  -----------")
variable_x1 = input_data[:,1] 
variable_y1 = input_data[:,8] 
plt.scatter(variable_x1, variable_y1) 
plt.show() 
# Scatter plot of 'TOEFL Score' vs 'Mark total 300'
print("------------- Scatter Plot:'TOEFL Score' vs 'Chance of Admit ' -----------")

variable_x2 = input_data[:, 2] 
variable_y2 = input_data[:, 8] 
plt.scatter(variable_x2, variable_y2) 
plt.show() 

variable_x3 = input_data[:, 3] 
variable_y3 = input_data[:, 8] 
plt.scatter(variable_x3, variable_y3) 
plt.show()

variable_x4 = input_data[:, 6] 
variable_y4 = input_data[:, 8] 
plt.scatter(variable_x4, variable_y4) 
plt.show()
# basic plot
dataframe1.boxplot()
dataframe1.hist()
dataframe1['Chance of Admit '].value_counts().head(10).plot.pie()

# Unsquish the pie.
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal')