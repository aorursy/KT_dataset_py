from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook as tqdm
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
nRowsRead = 1000 # specify 'None' if want to read whole file

# animeListGenres.csv has 15729 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/animeListGenres.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'animeListGenres.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
nRowsRead = 1000 # specify 'None' if want to read whole file

# animeReviewsOrderByTime.csv has 135201 rows in reality, but we are only loading/previewing the first 1000 rows





with open('../input/animeReviewsOrderByTime.csv', 'r', encoding='utf-8') as f:

    headers = f.readline().replace('"','').replace('\n','').split(',')

    print(headers)

    print('The number of column: ', len(headers))

    dataFormat = dict()

    for header in headers:

        dataFormat[header] = list()



    for idx, line in enumerate(tqdm(f.readlines(), desc='Now parsing... ')):

        

        if idx == 67:

            yee = line

        

        if line != '':

            line = line.replace('\n','')

            indices = [i for i, x in enumerate(line) if x == ',']

            idxStart = 0

            for i in range(len(headers)):

                if i < len(headers) - 1:

                    dataFormat[headers[i]].append(line[idxStart + 1:indices[i] - 1])

                    idxStart = indices[i] + 1

                elif i == len(headers) - 1:

                    dataFormat[headers[i]].append(line[idxStart + 1:-1])

                else:

                    break

        if nRowsRead is not None and nRowsRead == idx + 1:

            print('We read only', nRowsRead, 'lines.')

            break
df2 = pd.DataFrame(dataFormat)

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)