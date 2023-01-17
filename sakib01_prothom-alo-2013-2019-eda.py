from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))
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

nRowsRead = 1000 # specify 'None' if want to read whole file

# content_2013.csv has 59945 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/content_2013.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'content_2013.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 6, 15)
nRowsRead = 1000 # specify 'None' if want to read whole file

# content_2014.csv has 97335 rows in reality, but we are only loading/previewing the first 1000 rows

df2 = pd.read_csv('../input/content_2014.csv', delimiter=',', nrows = nRowsRead)

df2.dataframeName = 'content_2014.csv'

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 6, 15)
nRowsRead = 1000 # specify 'None' if want to read whole file

# content_2015.csv has 102637 rows in reality, but we are only loading/previewing the first 1000 rows

df3 = pd.read_csv('../input/content_2015.csv', delimiter=',', nrows = nRowsRead)

df3.dataframeName = 'content_2015.csv'

nRow, nCol = df3.shape

print(f'There are {nRow} rows and {nCol} columns')
df3.head(5)
plotPerColumnDistribution(df3, 10, 5)
plotCorrelationMatrix(df3, 8)
plotScatterMatrix(df3, 6, 15)
df1 = pd.read_csv('../input/content_2013.csv', delimiter=',')

df2 = pd.read_csv('../input/content_2014.csv', delimiter=',')

df3 = pd.read_csv('../input/content_2015.csv', delimiter=',')

df4 = pd.read_csv('../input/content_2016.csv', delimiter=',')

df5 = pd.read_csv('../input/content_2017.csv', delimiter=',')

df6 = pd.read_csv('../input/content_2018.csv', delimiter=',')

df7 = pd.read_csv('../input/content_2019.csv', delimiter=',')

df_total = pd.concat([df1, df2, df3, df4, df5, df6, df7], axis=0)
#df_total 
df_total  = df_total.reset_index(drop = True)
#df_total
df_total['rape_news'] = df_total['title'].str.contains('ধর্ষণ|ধর্ষন', regex = True).map({True:1, False:0})
df_total[df_total['rape_news'] == 1]
#(df_total.find('ধর্ষণ') != -1 or df_total.find('ধর্ষন') != -1)
df_total['date'] = pd.to_datetime(df_total['date'])


df_total['weekday_name'] = [d.strftime('%a') for d in df_total['date']]
df_total['week_number'] = [str(d.week) + " " + str(d.year) for d in df_total['date']]
df_total['month_number'] = [str(d.month) + " " + str(d.year) for d in df_total['date']]
df_total.head()
import matplotlib.pyplot as plt



plt.plot(df_total.groupby('weekday_name')['rape_news'].sum())
df_total.groupby('weekday_name')['rape_news'].sum()
plt.figure(figsize=(20,10))

plt.plot(df_total.groupby('week_number', sort = False)['rape_news'].sum())
plt.figure(figsize=(20,10))

plt.plot(df_total.groupby('week_number', sort = False)['rape_news'].sum()[:20])
temp = df_total.groupby('week_number', sort = False)['rape_news'].sum()

#temp[temp['rape_news'] == temp['rape_news'].max()]

plt.figure(figsize=(20,10))

plt.plot(df_total.groupby('date', sort = False)['rape_news'].sum())
plt.figure(figsize=(20,10))

plt.plot(df_total.groupby('month_number', sort = False)['rape_news'].sum())
plt.figure(figsize=(20,10))

plt.plot(df_total.groupby('month_number', sort = False)['rape_news'].sum()[:20])
#df_total.groupby('month_number', sort = False)['rape_news'].sum()
#df_total[(df_total['month_number'] == '5 2017') & (df_total['rape_news'] == 1)]
import seaborn as sns

sns.countplot(x = df_total['rape_news'])
grouping = df_total.groupby('month_number', sort = False)[['rape_news']].sum()
grouping.head()
sns.scatterplot(grouping.index,grouping ['rape_news'])
import plotly.offline as py

import plotly.graph_objs as go
date_group = df_total.groupby('date', sort = False)[['rape_news']].sum()
date_group.head()
trace0 = go.Scatter(x=date_group.index, y=date_group['rape_news'])

data = [trace0]

py.iplot(data)