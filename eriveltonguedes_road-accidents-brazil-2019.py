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

# brazil-2019.csv has 15588 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/brazil-2019.csv', delimiter=';', nrows = nRowsRead, sep=',', encoding='1252')

# help (pd.read_csv)

df1.dataframeName = 'brazil-2019.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')

df1.tail()
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import folium



# Create a Map instance

m = folium.Map(location=[60.25, 24.8],

    zoom_start=10, control_scale=True)

m

# Let's change the basemap style to 'Stamen Toner'

m = folium.Map(location=[40.730610, -73.935242], tiles='Stamen Toner',

                zoom_start=12, control_scale=True, prefer_canvas=True)



m
# Create a Map instance

m = folium.Map(location=[60.20, 24.96],

    zoom_start=12, control_scale=True)



# Add marker

# Run: help(folium.Icon) for more info about icons

folium.Marker(

    location=[60.20426, 24.96179],

    popup='Kumpula Campus',

    icon=folium.Icon(color='green', icon='ok-sign'),

).add_to(m)



#Show map

m
# Create a Map instance

m = folium.Map(location=[-19.90, -44.02],

    zoom_start=12, control_scale=True)



# Add marker

# Run: help(folium.Icon) for more info about icons

for r in df1:

    folium.Marker(

    #location=[df1.latitude,df1.longitude],

    location=[-19.80, -43.02],    

    popup='Radar 01',

    icon=folium.Icon(color='green', icon='ok-sign'),).add_to(m)





m





folium.Marker(

    location=[-19.9577068226568, -44.0299821167655],

    popup='Radar 01',

    icon=folium.Icon(color='green', icon='ok-sign'),

).add_to(m)

folium.Marker(

    location=[-19.961545578148, -44.0383238588943],

    popup='Radar 02',

    icon=folium.Icon(color='green', icon='ok-sign'),

).add_to(m)

folium.Marker(

    location=[-20.0051375297523, -44.2139376946762],

    popup='Radar 03',

    icon=folium.Icon(color='green', icon='ok-sign'),

).add_to(m)



#Show map

m