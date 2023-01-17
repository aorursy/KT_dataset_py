from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
nRowsRead = 1000 # specify 'None' if want to read whole file
# apartment_data.csv has 1124 rows in reality, but we are only loading/previewing the first 1000 rows


df = pd.read_csv('../input/apartment_data.csv',delimiter=',',nrows=nRowsRead)

nRow, nCol = df.shape

#df1.dataframeName = 'apartment_data.csv'
print(f'There are {nRow} rows and {nCol} columns')


df.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 6, 15)
df2 = pd.read_csv('../input/blore_apartment_data.csv', delimiter=',',nrows=1000)

nRow, nCol = df2.shape

print(f'There are {nRow} rows, {nCol} columns in the data')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)