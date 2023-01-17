# https://github.com/tanmoyie/Applied-Statistics

import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import plotly.plotly as plty
import plotly.graph_objs as go
from sklearn import datasets, linear_model
dataframe1 = pd.read_csv("../input/Grading of the students in the exam (OR).csv")# Grading of the students in the exam (IPE101) raw.csv 
print(dataframe1)
print(dataframe1.dtypes)
input_data = dataframe1.values
input_data_asMatrix = dataframe1.as_matrix()

# Examine the properties of the dataset
# print the statistics
print(dataframe1['Final Mark'].describe())
dataframe1.iloc[29,[1,2,3,4,5,6,7,8,9,10]] = 29 
#By list method 

print(dataframe1.values)
# Develop a Scatter plot of roll vs total class test
print("-------------Draw a Scatter Plot: roll vs total class test -----------")
x = input_data[:,9] # my roll is 29
y = input_data[:,5] # total class test
plt.scatter(x,y) 
plt.show()
# Develop a Scatter plot of attendace vs total marks
print("-------------Draw a Scatter Plot: attendace vs total marks -----------")
#y_final_exam_sum = input_data['']+input_data[:,11]
x_ct = input_data[:,6] # attendance 
y_final_exam = input_data[:,11] # total marks
plt.scatter(x_ct, y_final_exam) # Scatter plot for Attendance vs total marks
plt.show() 
# basic plot
dataframe1.boxplot()
# https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.hist.html
dataframe1.hist()