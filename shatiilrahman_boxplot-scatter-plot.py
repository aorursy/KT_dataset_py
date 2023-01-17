import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
# ploting related libraries
import matplotlib.pyplot as plt
import plotly.plotly as plty
import plotly.graph_objs as go
#modeling
from sklearn import datasets, linear_model
# load the EXCEL file & read the data 
dataframe = pd.read_csv("../input/Cgpa_Gre_Ielts_Toefil.csv")
print(dataframe) 
input_data = dataframe.values
dataframe.boxplot('gre_total')
x=input_data[:,0]
y=input_data[:,5]
plt.scatter(x,y)
plt.xlabel('CGPA')
plt.ylabel('Toefil_Score')
plt.show()