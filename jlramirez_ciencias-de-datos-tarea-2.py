from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
dfStudents = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

dfStudents.dataframeName = 'StudentsPerformance.csv'
rows, cols = dfStudents.shape

print(f'{rows} observaciones y {cols} características')
dfStudents.head(10)
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]]

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('Cantidad de observaciones')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (columna {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
plotPerColumnDistribution(dfStudents, 10, 5)
def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.show()

plotCorrelationMatrix(dfStudents, 8)
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

plotScatterMatrix(dfStudents, 9, 10)
dfStudents['total score'] = dfStudents['math score']+dfStudents['reading score']+dfStudents['writing score']
dfStudents.head(5)
dfStudents = pd.get_dummies(dfStudents, columns=['gender'])

dfStudents = pd.get_dummies(dfStudents, columns=['test preparation course'])

dfStudents = pd.get_dummies(dfStudents, columns=['race/ethnicity'])

dfStudents = pd.get_dummies(dfStudents, columns=['parental level of education'])

dfStudents = pd.get_dummies(dfStudents, columns=['lunch'])
dfStudents.head()
from sklearn.decomposition import PCA
dfSatander = pd.read_csv('../input/santander-customer-satisfaction/train.csv')
rows, cols = dfSatander.shape

print(f'{rows} observaciones y {cols} características')
reducer = PCA(n_components=10)

dfSantanderReduced = reducer.fit_transform(dfSatander)

rows, cols = dfSantanderReduced.shape

print(f'{rows} observaciones y {cols} características')
reducer.explained_variance_ratio_.sum()