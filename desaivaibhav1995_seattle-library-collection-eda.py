import numpy as np

import pandas as pd 



from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

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

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Correlation matrix

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

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
#Scatter and density plots

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
nRowsRead = 1000 # specify 'None' if want to read whole file

# library-collection-inventory.csv has 26817320 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/library-collection-inventory.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'library-collection-inventory.csv'

nRow, nCol = df1.shape
df1.head()
df1.isnull().sum()
row = df1.isnull().any(axis = 1)

column = df1.isnull().any(axis = 0)

df1.loc[row,column]
df1[df1['Title'].isnull()].index.tolist()
df1.drop(index = [64, 68, 97, 129, 149, 241, 466, 628, 787, 798, 902, 981], inplace = True)


df1['FloatingItem'].fillna('Not Applicable', inplace = True)



df1['Publisher'].fillna('Unknown', inplace = True)

df1['Author'].fillna('Unknown', inplace = True)

df1['Subjects'].fillna('Unknown', inplace = True)



df1['ISBN'].fillna(0, inplace = True)

df1['PublicationYear'].fillna(0, inplace = True)

df1.isnull().sum()
print('Item collections : {}'.format(df1['ItemCollection'].nunique()))

print('Item types : {}'.format(df1['ItemType'].nunique()))

print('Report Dates(unique) : {}'.format(df1['ReportDate'].nunique()))
df_2014 = df1.ix[df1['PublicationYear'] == '2014.',[0,1,5]] 

df_2014
print('Books Published in 2014 : {}'.format(df_2014['BibNum'].count()))

print('Unique publishers of books in 2014 : {}'.format(df_2014['Publisher'].count()))
df_2016 = df1.ix[df1['PublicationYear'] == '2016.',[0,1,5]] 

df_2016
print('Books Published in 2016 : {}'.format(df_2016['BibNum'].count()))

print('Unique publishers of books in 2016 : {}'.format(df_2016['Publisher'].count()))
df1['ItemType'].value_counts()
adult_book = df1.ix[df1['ItemType'] == 'acbk',[1,2,4,5,6,8]]

adult_book
adult_book['PublicationYear'].value_counts().sort_values(ascending = False).head(10)
adult_book['ItemCollection'].value_counts().head(10)
adult_book['Publisher'].value_counts().head()
adult_book['Subjects'].value_counts().head()
print('Total books in the category : {}'.format(adult_book['Title'].count()))

print('Unique books in this category : {}'.format(adult_book['Title'].nunique()))
adult_book['Author'].value_counts().head()
adult_book.loc[adult_book['Author'] == 'Chicago, Judy, 1939-']
adult_book.loc[adult_book['Author'] == 'Patterson, James, 1947-']