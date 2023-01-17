import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
import os
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 9999
pd.options.display.float_format = '{:20,.2f}'.format
DataSet = pd.read_csv('../input/asiangamestop10.csv').fillna(0)
TotalRowCount = len(DataSet)
print("Total Number of Data Count :", TotalRowCount)
DataSet.dtypes
DataSet.head(10)
DataSet.rename(columns={'NOC' : 'Country',}, inplace=True)
DataSet.head(10)
TotalPrice = DataSet.groupby(['Year'])['Total'].sum().nlargest(15)
print("Top 15 Year Wise Medals Count\n")
print(TotalPrice)
plt.figure(figsize=(22,7))
GraphData=DataSet.groupby(['Year'])['Total'].sum().nlargest(15)
GraphData.plot(kind='bar')
plt.ylabel('Medals Count')
plt.xlabel('Year')
TotalPrice = DataSet.groupby(['Country'])['Total'].sum().nlargest(15)
print("Top 15 Country Medals Count \n")
print(TotalPrice)
plt.figure(figsize=(22,7))
GraphData=DataSet.groupby(['Country'])['Total'].sum().nlargest(15)
GraphData.plot(kind='bar')
plt.ylabel('Medals Count')
plt.xlabel('Country Name')
UniqueNOC = DataSet['Country'].unique()
print("All Unique Country Name \n")
print(UniqueNOC)
print ("Max Medal Mode is :",DataSet['Total'].max())
print ("Min Medal Mode is :",DataSet['Total'].min())
ItemTypeMean = DataSet['Total'].mean()
print ("Mean Medal Mode is :", round(ItemTypeMean))
ItemData=DataSet[DataSet['Country']=='China (CHN)']
print ("China (CHN) - Max Medal Mode is :",ItemData['Total'].max())
print ("China (CHN) - Min Medal Mode is :",ItemData['Total'].min())
ItemTypeMean = ItemData['Total'].mean()
print ("China (CHN) - Mean Medal Mode is :", round(ItemTypeMean))