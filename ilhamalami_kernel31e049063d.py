# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#nRowsRead = 1000 # specify 'None' if want to read whole file

# webPhishing.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/ksahealth/Main-dataset.csv', delimiter=',')

df1.dataframeName = 'Main-dataset.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1.tail()
df1 = df1.drop('Unnamed: 0', axis=1)
# Distribution graphs (histogram/bar graph) of column data



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
plotPerColumnDistribution(df1, 12, 4)
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    #filename = df.dataframeName

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

plotCorrelationMatrix(df1, 8)
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
plotScatterMatrix(df1, 20, 10)
# Fixing the data types

df1['val'] = df1['val'].replace('.', '')

df1['val'] = df1['val'].astype(int)
df1.head()
dataset = df1.groupby(['cause'], as_index=False).mean()

causes = df1.cause.unique()

dataset = pd.DataFrame({'cause': causes,

                        'estimated_measure':dataset['val'],

                        'Upper_Bound_Uncertainty':dataset['upper'],

                        'Lower_Bound_Uncertainty':dataset['lower'], 

                        })

dataset = dataset.sort_values(by='estimated_measure', ascending=False)

#dataset['location'] = dataset['location'].str.replace('Saudi Arabia', 'Saudi')

dataset.head()
df_top_5 = df1.loc[df1['cause'].isin(dataset['cause'][:5])]

df_top_5 = df_top_5.reset_index().drop(columns='index')

df_top_5 = df_top_5.groupby(['cause', 'year'], as_index=False).sum()
import seaborn as sns

plt.style.use('seaborn-ticks')



fig, ax = plt.subplots(figsize=(18, 8))



ax = sns.lineplot(x='year', y='val', hue='cause', data=df_top_5, palette='colorblind')



plt.grid(color='k', axis='both', alpha=0.3, lw=0.8)

plt.tick_params(axis='x', labelrotation=30, length=6)

plt.legend(loc='best', fontsize='medium', frameon=True)

plt.ylabel('val', fontsize=12)

plt.xlabel('Year', fontsize=12)

plt.title('estimated_measure for the top 5 cause', fontsize=20)

plt.xticks(np.arange(2007,2017))

plt.xlim(2007,2017)

plt.show()
print(df1.isnull().sum())
table = df1.groupby(['year', 'sex'], as_index = False).sum()





plt.style.use('seaborn-ticks')

fig, ax = plt.subplots(figsize=(18, 8))



ax = sns.lineplot(x='year', y='val', hue='sex', data=table)



plt.grid(color='k', axis='both', alpha=0.3, lw=0.8)

plt.tick_params(axis='x', labelrotation=30, length=6)

plt.legend(loc='best', fontsize='medium', frameon=True)

plt.ylabel('val', fontsize=12)

plt.xlabel('Year', fontsize=12)

plt.title('causes', fontsize=20)

plt.xticks(np.arange(2006,2018))

plt.xlim(2006,2018)

plt.show()