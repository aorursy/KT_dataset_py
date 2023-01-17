import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# There is 1 csv file in the current version of the dataset:
print(os.listdir('../input'))
# Plot the PCA with either 2 or 3 reduced components
def plotPCA(df, nComponents):
	df = df.select_dtypes(include =[np.number]) # keep only numerical columns
	df = df.dropna('columns') # drop columns with NaN
	if df.shape[1] < nComponents:
		print(f'The number of numeric columns ({df.shape[1]}) is less than the number of PCA components ({nComponents})')
		return
	df = df.astype('float64') # Cast to float for sklearn functions
	df = StandardScaler().fit_transform(df) # Standardize features by removing the mean and scaling to unit variance
	pca = PCA(n_components = nComponents)
	principalComponents = pca.fit_transform(df)
	principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component ' + str(i) for i in range(1, nComponents + 1)])
	fig = plt.figure(figsize = (8, 8))
	if (nComponents == 3):
		ax = fig.add_subplot(111, projection = '3d')
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_zlabel('Principal Component 3', fontsize = 15)
		ax.set_title('3 component PCA', fontsize = 20)
		ax.scatter(xs = principalDf.iloc[:, 0], ys = principalDf.iloc[:, 1], zs = principalDf.iloc[:, 2])
	else:
		ax = fig.add_subplot(111)
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_title('2 component PCA', fontsize = 20)
		ax.scatter(x = principalDf.iloc[:, 0], y = principalDf.iloc[:, 1])

# Histogram of column data
def plotHistogram(df, nHistogramShown, nHistogramPerRow):
	nRow, nCol = df.shape
	columnNames = list(df)
	nHistRow = (nCol + nHistogramPerRow - 1) / nHistogramPerRow
	plt.figure(num=None, figsize=(6*nHistogramPerRow, 5*nHistRow), dpi=80, facecolor='w', edgecolor='k')
	for i in range(min(nCol, nHistogramShown)):
		plt.subplot(nHistRow, nHistogramPerRow, i+1)
		df.iloc[:,i].hist()
		plt.ylabel('counts')
		plt.title(f'{columnNames[i]} (column {i})')
	plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
	plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
	corr = df.corr()
	corrMat = plt.matshow(corr, fignum = 1)
	plt.xticks(range(len(corr.columns)), corr.columns)
	plt.yticks(range(len(corr.columns)), corr.columns)
	plt.colorbar(corrMat)
	plt.title(f'Correlation Matrix for {df.name}')
	plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
	df = df.select_dtypes(include =[np.number]) # keep only numerical columns
	# Remove rows and columns that would lead to df being singular
	df = df.dropna('columns')
	df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
	columnNames = list(df)
	if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
		columnNames = columnNames[:10]
	df = df[columnNames]
	ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
	corrs = df.corr().values
	for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
		ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
	plt.suptitle('Scatter and Density Plot')
	plt.show()

nRowsRead = 100 # specify 'None' if want to read whole file
df0 = pd.read_csv('../input/benhamner.csv', delimiter=',', nrows = nRowsRead)
df0.name = 'benhamner.csv'
# benhamner.csv has 2702 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df0.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df0)

# Histogram of sampled columns
plotHistogram(df0, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df0, 8)

# Scatter and density plots
plotScatterMatrix(df0, 15, 6)

# PCA Analysis
plotPCA(df0, 2) # 2D PCA
plotPCA(df0, 3) # 3D PCA
