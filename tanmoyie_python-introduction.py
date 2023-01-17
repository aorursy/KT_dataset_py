# useful link
# https://github.com/tanmoyie/Applied-Statistics
# we use hash (#) to make single line comment, triple inverted comma (""" """) to make multiple line comment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
# ploting related libraries
import matplotlib.pyplot as plt
import plotly.plotly as plty
import plotly.graph_objs as go
# modeling
from sklearn import datasets, linear_model
# load the EXCEL file & read the data 
#dataframe_clean = pd.read_csv("../input/grading-of-the-students-in-the-exam-ipe101-raw.csv")

dataframe_raw = pd.read_csv("../input/Grading of the students in the exam (IPE101) raw.csv", header=0).drop_duplicates()
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html

print("----------dataframe_raw")
print(dataframe_raw)
print("-------------dataframe_raw.dtypes")
print(dataframe_raw.dtypes)

# finding NA values
df_droping_na = dataframe_raw.dropna()
input_data = df_droping_na.values
type(input_data)
print("-----------input_data")
print(input_data)
#what is the data format of data_array
