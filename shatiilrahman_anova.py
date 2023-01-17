import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import plotly.plotly as plty
import plotly.graph_objs as go
from sklearn import datasets, linear_model
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import OLS
dataframe1 = pd.read_csv("../input/Untitled-Spreadsheet_Sheet1.csv")
print(dataframe1)
input_data = dataframe1.values
print(dataframe1['Year 1 Scores'].describe())
print(dataframe1['Year 2 Scores'].describe())
print(dataframe1['Year 3 Scores'].describe())
from pandas.tools import plotting
plotting.scatter_matrix(dataframe1[['Year 1 Scores', 'Year 2 Scores', 'Year 3 Scores']])   
dataframe1.boxplot()
dataframe1.hist()

x=input_data[:,0]
y=input_data[:,1]
z=input_data[:,2]
model = sm.OLS(x,y,z)
results = model.fit()
print(results.summary()) 
