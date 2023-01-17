import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# There are 31 csv files in the current version of the dataset:
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
df0 = pd.read_csv('../input/Competitions.csv', delimiter=',', nrows = nRowsRead)
df0.name = 'Competitions.csv'
# Competitions.csv has 919 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df0.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df0)

# Histogram of sampled columns
plotHistogram(df0, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df0, 8)

# Scatter and density plots
plotScatterMatrix(df0, 20, 5)

# PCA Analysis
plotPCA(df0, 2) # 2D PCA
plotPCA(df0, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df1 = pd.read_csv('../input/CompetitionTags.csv', delimiter=',', nrows = nRowsRead)
df1.name = 'CompetitionTags.csv'
# CompetitionTags.csv has 311 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df1)

# Histogram of sampled columns
plotHistogram(df1, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df1, 8)

# Scatter and density plots
plotScatterMatrix(df1, 9, 10)

# PCA Analysis
plotPCA(df1, 2) # 2D PCA
plotPCA(df1, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df2 = pd.read_csv('../input/Datasets.csv', delimiter=',', nrows = nRowsRead)
df2.name = 'Datasets.csv'
# Datasets.csv has 8950 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df2)

# Histogram of sampled columns
plotHistogram(df2, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df2, 8)

# Scatter and density plots
plotScatterMatrix(df2, 20, 5)

# PCA Analysis
plotPCA(df2, 2) # 2D PCA
plotPCA(df2, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df3 = pd.read_csv('../input/DatasetTags.csv', delimiter=',', nrows = nRowsRead)
df3.name = 'DatasetTags.csv'
# DatasetTags.csv has 6943 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df3)

# Histogram of sampled columns
plotHistogram(df3, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df3, 8)

# Scatter and density plots
plotScatterMatrix(df3, 9, 10)

# PCA Analysis
plotPCA(df3, 2) # 2D PCA
plotPCA(df3, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df4 = pd.read_csv('../input/DatasetVersions.csv', delimiter=',', nrows = nRowsRead)
df4.name = 'DatasetVersions.csv'
# DatasetVersions.csv has 23255 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df4.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df4)

# Histogram of sampled columns
plotHistogram(df4, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df4, 8)

# Scatter and density plots
plotScatterMatrix(df4, 20, 5)

# PCA Analysis
plotPCA(df4, 2) # 2D PCA
plotPCA(df4, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df5 = pd.read_csv('../input/DatasetVotes.csv', delimiter=',', nrows = nRowsRead)
df5.name = 'DatasetVotes.csv'
# DatasetVotes.csv has 78418 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df5.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df5)

# Histogram of sampled columns
plotHistogram(df5, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df5, 8)

# Scatter and density plots
plotScatterMatrix(df5, 9, 10)

# PCA Analysis
plotPCA(df5, 2) # 2D PCA
plotPCA(df5, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df6 = pd.read_csv('../input/DatasourceObjects.csv', delimiter=',', nrows = nRowsRead)
df6.name = 'DatasourceObjects.csv'
# DatasourceObjects.csv has 88257 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df6.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df6)

# Histogram of sampled columns
plotHistogram(df6, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df6, 8)

# Scatter and density plots
plotScatterMatrix(df6, 12, 7)

# PCA Analysis
plotPCA(df6, 2) # 2D PCA
plotPCA(df6, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df7 = pd.read_csv('../input/Datasources.csv', delimiter=',', nrows = nRowsRead)
df7.name = 'Datasources.csv'
# Datasources.csv has 8850 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df7.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df7)

# Histogram of sampled columns
plotHistogram(df7, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df7, 8)

# Scatter and density plots
plotScatterMatrix(df7, 9, 10)

# PCA Analysis
plotPCA(df7, 2) # 2D PCA
plotPCA(df7, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df8 = pd.read_csv('../input/DatasourceVersionObjectTables.csv', delimiter=',', nrows = nRowsRead)
df8.name = 'DatasourceVersionObjectTables.csv'
# DatasourceVersionObjectTables.csv has 1721 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df8.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df8)

# Histogram of sampled columns
plotHistogram(df8, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df8, 8)

# Scatter and density plots
plotScatterMatrix(df8, 9, 10)

# PCA Analysis
plotPCA(df8, 2) # 2D PCA
plotPCA(df8, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df9 = pd.read_csv('../input/ForumMessages.csv', delimiter=',', nrows = nRowsRead)
df9.name = 'ForumMessages.csv'
# ForumMessages.csv has 310745 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df9.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df9)

# Histogram of sampled columns
plotHistogram(df9, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df9, 8)

# Scatter and density plots
plotScatterMatrix(df9, 15, 6)

# PCA Analysis
plotPCA(df9, 2) # 2D PCA
plotPCA(df9, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df10 = pd.read_csv('../input/ForumMessageVotes.csv', delimiter=',', nrows = nRowsRead)
df10.name = 'ForumMessageVotes.csv'
# ForumMessageVotes.csv has 322911 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df10.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df10)

# Histogram of sampled columns
plotHistogram(df10, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df10, 8)

# Scatter and density plots
plotScatterMatrix(df10, 12, 7)

# PCA Analysis
plotPCA(df10, 2) # 2D PCA
plotPCA(df10, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df11 = pd.read_csv('../input/Forums.csv', delimiter=',', nrows = nRowsRead)
df11.name = 'Forums.csv'
# Forums.csv has 11431 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df11.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df11)

# Histogram of sampled columns
plotHistogram(df11, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df11, 8)

# Scatter and density plots
plotScatterMatrix(df11, 6, 15)

# PCA Analysis
plotPCA(df11, 2) # 2D PCA
plotPCA(df11, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df12 = pd.read_csv('../input/ForumTopics.csv', delimiter=',', nrows = nRowsRead)
df12.name = 'ForumTopics.csv'
# ForumTopics.csv has 49635 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df12.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df12)

# Histogram of sampled columns
plotHistogram(df12, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df12, 8)

# Scatter and density plots
plotScatterMatrix(df12, 20, 5)

# PCA Analysis
plotPCA(df12, 2) # 2D PCA
plotPCA(df12, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df13 = pd.read_csv('../input/KernelLanguages.csv', delimiter=',', nrows = nRowsRead)
df13.name = 'KernelLanguages.csv'
nRow, nCol = df13.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df13)

# Histogram of sampled columns
plotHistogram(df13, 10, 5)

# PCA Analysis
plotPCA(df13, 2) # 2D PCA
plotPCA(df13, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df14 = pd.read_csv('../input/Kernels.csv', delimiter=',', nrows = nRowsRead)
df14.name = 'Kernels.csv'
# Kernels.csv has 183260 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df14.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df14)

# Histogram of sampled columns
plotHistogram(df14, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df14, 8)

# Scatter and density plots
plotScatterMatrix(df14, 20, 5)

# PCA Analysis
plotPCA(df14, 2) # 2D PCA
plotPCA(df14, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df15 = pd.read_csv('../input/KernelTags.csv', delimiter=',', nrows = nRowsRead)
df15.name = 'KernelTags.csv'
# KernelTags.csv has 18316 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df15.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df15)

# Histogram of sampled columns
plotHistogram(df15, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df15, 8)

# Scatter and density plots
plotScatterMatrix(df15, 9, 10)

# PCA Analysis
plotPCA(df15, 2) # 2D PCA
plotPCA(df15, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df16 = pd.read_csv('../input/KernelVersionCompetitionSources.csv', delimiter=',', nrows = nRowsRead)
df16.name = 'KernelVersionCompetitionSources.csv'
# KernelVersionCompetitionSources.csv has 309564 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df16.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df16)

# Histogram of sampled columns
plotHistogram(df16, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df16, 8)

# Scatter and density plots
plotScatterMatrix(df16, 9, 10)

# PCA Analysis
plotPCA(df16, 2) # 2D PCA
plotPCA(df16, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df17 = pd.read_csv('../input/KernelVersionDatasetSources.csv', delimiter=',', nrows = nRowsRead)
df17.name = 'KernelVersionDatasetSources.csv'
# KernelVersionDatasetSources.csv has 447746 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df17.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df17)

# Histogram of sampled columns
plotHistogram(df17, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df17, 8)

# Scatter and density plots
plotScatterMatrix(df17, 9, 10)

# PCA Analysis
plotPCA(df17, 2) # 2D PCA
plotPCA(df17, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df18 = pd.read_csv('../input/KernelVersionKernelSources.csv', delimiter=',', nrows = nRowsRead)
df18.name = 'KernelVersionKernelSources.csv'
# KernelVersionKernelSources.csv has 15121 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df18.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df18)

# Histogram of sampled columns
plotHistogram(df18, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df18, 8)

# Scatter and density plots
plotScatterMatrix(df18, 9, 10)

# PCA Analysis
plotPCA(df18, 2) # 2D PCA
plotPCA(df18, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df19 = pd.read_csv('../input/KernelVersionOutputFiles.csv', delimiter=',', nrows = nRowsRead)
df19.name = 'KernelVersionOutputFiles.csv'
# KernelVersionOutputFiles.csv has 4929112 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df19.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df19)

# Histogram of sampled columns
plotHistogram(df19, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df19, 8)

# Scatter and density plots
plotScatterMatrix(df19, 9, 10)

# PCA Analysis
plotPCA(df19, 2) # 2D PCA
plotPCA(df19, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df20 = pd.read_csv('../input/KernelVersions.csv', delimiter=',', nrows = nRowsRead)
df20.name = 'KernelVersions.csv'
# KernelVersions.csv has 990238 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df20.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df20)

# Histogram of sampled columns
plotHistogram(df20, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df20, 8)

# Scatter and density plots
plotScatterMatrix(df20, 20, 5)

# PCA Analysis
plotPCA(df20, 2) # 2D PCA
plotPCA(df20, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df21 = pd.read_csv('../input/KernelVotes.csv', delimiter=',', nrows = nRowsRead)
df21.name = 'KernelVotes.csv'
# KernelVotes.csv has 276209 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df21.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df21)

# Histogram of sampled columns
plotHistogram(df21, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df21, 8)

# Scatter and density plots
plotScatterMatrix(df21, 9, 10)

# PCA Analysis
plotPCA(df21, 2) # 2D PCA
plotPCA(df21, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df22 = pd.read_csv('../input/Organizations.csv', delimiter=',', nrows = nRowsRead)
df22.name = 'Organizations.csv'
# Organizations.csv has 1764 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df22.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df22)

# Histogram of sampled columns
plotHistogram(df22, 10, 5)

# PCA Analysis
plotPCA(df22, 2) # 2D PCA
plotPCA(df22, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df23 = pd.read_csv('../input/Submissions.csv', delimiter=',', nrows = nRowsRead)
df23.name = 'Submissions.csv'
# Submissions.csv has 3780936 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df23.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df23)

# Histogram of sampled columns
plotHistogram(df23, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df23, 8)

# Scatter and density plots
plotScatterMatrix(df23, 20, 5)

# PCA Analysis
plotPCA(df23, 2) # 2D PCA
plotPCA(df23, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df24 = pd.read_csv('../input/Tags.csv', delimiter=',', nrows = nRowsRead)
df24.name = 'Tags.csv'
# Tags.csv has 574 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df24.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df24)

# Histogram of sampled columns
plotHistogram(df24, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df24, 8)

# Scatter and density plots
plotScatterMatrix(df24, 15, 6)

# PCA Analysis
plotPCA(df24, 2) # 2D PCA
plotPCA(df24, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df25 = pd.read_csv('../input/TeamMemberships.csv', delimiter=',', nrows = nRowsRead)
df25.name = 'TeamMemberships.csv'
# TeamMemberships.csv has 1045896 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df25.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df25)

# Histogram of sampled columns
plotHistogram(df25, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df25, 8)

# Scatter and density plots
plotScatterMatrix(df25, 9, 10)

# PCA Analysis
plotPCA(df25, 2) # 2D PCA
plotPCA(df25, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df26 = pd.read_csv('../input/Teams.csv', delimiter=',', nrows = nRowsRead)
df26.name = 'Teams.csv'
# Teams.csv has 1025387 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df26.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df26)

# Histogram of sampled columns
plotHistogram(df26, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df26, 8)

# Scatter and density plots
plotScatterMatrix(df26, 20, 5)

# PCA Analysis
plotPCA(df26, 2) # 2D PCA
plotPCA(df26, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df27 = pd.read_csv('../input/UserAchievements.csv', delimiter=',', nrows = nRowsRead)
df27.name = 'UserAchievements.csv'
# UserAchievements.csv has 5768400 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df27.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df27)

# Histogram of sampled columns
plotHistogram(df27, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df27, 8)

# Scatter and density plots
plotScatterMatrix(df27, 20, 5)

# PCA Analysis
plotPCA(df27, 2) # 2D PCA
plotPCA(df27, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df28 = pd.read_csv('../input/UserFollowers.csv', delimiter=',', nrows = nRowsRead)
df28.name = 'UserFollowers.csv'
# UserFollowers.csv has 107307 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df28.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df28)

# Histogram of sampled columns
plotHistogram(df28, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df28, 8)

# Scatter and density plots
plotScatterMatrix(df28, 9, 10)

# PCA Analysis
plotPCA(df28, 2) # 2D PCA
plotPCA(df28, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df29 = pd.read_csv('../input/UserOrganizations.csv', delimiter=',', nrows = nRowsRead)
df29.name = 'UserOrganizations.csv'
# UserOrganizations.csv has 3375 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df29.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df29)

# Histogram of sampled columns
plotHistogram(df29, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df29, 8)

# Scatter and density plots
plotScatterMatrix(df29, 9, 10)

# PCA Analysis
plotPCA(df29, 2) # 2D PCA
plotPCA(df29, 3) # 3D PCA

nRowsRead = 100 # specify 'None' if want to read whole file
df30 = pd.read_csv('../input/Users.csv', delimiter=',', nrows = nRowsRead)
df30.name = 'Users.csv'
# Users.csv has 1922925 rows in reality, but we are only loading/previewing the first 100 rows
nRow, nCol = df30.shape
print(f'There are {nRow} rows and {nCol} columns')
columnNames = list(df30)

# Histogram of sampled columns
plotHistogram(df30, 10, 5)

# Correlation matrix
plotCorrelationMatrix(df30, 8)

# Scatter and density plots
plotScatterMatrix(df30, 6, 15)

# PCA Analysis
plotPCA(df30, 2) # 2D PCA
plotPCA(df30, 3) # 3D PCA
