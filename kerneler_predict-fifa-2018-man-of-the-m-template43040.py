import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import os
# There is 1 csv file in the current version of the dataset:
print(os.listdir('../input'))
nRowsRead = 100 # specify 'None' if want to read whole file
df0 = pd.read_csv('../input/FIFA 2018 Statistics.csv', delimiter=',', nrows = nRowsRead)

print('FIFA 2018 Statistics.csv has 128 rows in reality, but we are only loading/previewing the first 100 rows')
nRow, nCol = df0.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df0)

# Histogram of sampled columns
nHistShown = 10
nHistCol = 5
nHistRow = (nCol + nHistCol - 1) / nHistCol
plt.figure(num=None, figsize=(6*nHistCol, 5*nHistRow), dpi=80, facecolor='w', edgecolor='k')
for i in range(min(nCol, nHistShown)):
	plt.subplot(nHistRow, nHistCol, i+1)
	df0.iloc[:,i].hist()
	plt.ylabel('counts')
	plt.title(f'{columnNames[i]} (column {i})')
plt.show()

# Correlation matrix
plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
corr = df0.corr()
corrMat = plt.matshow(corr, fignum = 1)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(corrMat)
plt.title('Correlation Matrix for FIFA 2018 Statistics.csv')
plt.show()

# Scatter and density plots
numericColumns = df0[['Goal Scored', 'Ball Possession %', 'Attempts', 'On-Target', 'Off-Target', 'Blocked', 'Corners', 'Offsides', 'Free Kicks', 'Saves', 'Pass Accuracy %', 'Passes', 'Distance Covered (Kms)', 'Fouls Committed', 'Yellow Card', 'Yellow & Red', 'Red', '1st Goal', 'Goals in PSO', 'Own goals', 'Own goal Time']]
# Remove rows and columns that would lead to numericColumns being singular
numericColumns = numericColumns.dropna('columns')
numericColumns = numericColumns.loc[:, (numericColumns != 0).any(axis = 0)]
numericColumns = numericColumns.loc[(numericColumns != 0).any(axis = 1), :]
matrixPlotDimension = 20
axes = pd.plotting.scatter_matrix(numericColumns, alpha=0.75, figsize=[matrixPlotDimension, matrixPlotDimension], diagonal='kde')
plt.suptitle('Scatter and Density Plot')
plt.show()