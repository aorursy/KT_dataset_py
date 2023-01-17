import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/hotel LA1 - hotel LA1.csv")
pd.set_option('display.max_columns', None)  

pd.set_option('display.expand_frame_repr', False)

pd.set_option('max_colwidth', -1)

data.head()
data.shape
del data['Unnamed: 0']

del data['Unnamed: 1']
data.index
data.head()
import seaborn as sns

import matplotlib.pyplot as plt
data.describe()
data.head()
for col in data.columns: 

    print(col) 
#data.rename(columns = {'Unnamed: 2':'Wh'}, inplace = True) 

#data.rename(columns = {"line to line Voltage Red phase to yellow phase":"Voltage ry"}, inplace = True)

#data.rename(columns = {'line to line Voltage yellow phase to blue phase':'Voltage yb'}, inplace = True)

#data.rename(columns = {'line to line Voltage blue phase to red phase':'Voltage br'}, inplace = True)

#data.rename(columns = {'phase Voltage Red':'Voltage R'}, inplace = True)

#data.rename(columns = {'phase Voltage Yellow':'Voltage Y'}, inplace = True)

#data.rename(columns = {'phase Voltage blue':'Voltage B'}, inplace = True)

#data.rename(columns = {'Unnamed: 9':'Current R'}, inplace = True)

#data.rename(columns = {'Unnamed: 10':'Current Y'}, inplace = True)

#data.rename(columns = {'Unnamed: 11':'Current B'}, inplace = True)

#data.rename(index=str, columns={"line to line Voltage Red phase to yellow phase":"Voltage ry"})
new_header = data.iloc[0] #grab the first row for the header

data = data[1:] #take the data less the header row

data.columns = new_header
for col in data.columns: 

    print(col) 
data.head()
data.describe()
plt.plot(data.describe())