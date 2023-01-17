# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

df = pd.read_csv('../input/cusersmarildownloadsdeathscsv/deaths.csv', delimiter=';', encoding = "ISO-8859-1", nrows = nRowsRead)

df.dataframeName = 'deaths.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
df.describe()
# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in df.columns if

                    df[cname].nunique() < 10 and 

                    df[cname].dtype == "object"]





# Select numerical columns

numerical_cols = [cname for cname in df.columns if 

                df[cname].dtype in ['int64', 'float64']]
print(categorical_cols)
print(numerical_cols)
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
plotPerColumnDistribution(df, 10, 5)
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='r', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
plotCorrelationMatrix(df, 8)
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
plotScatterMatrix(df, 8, 6)
print ("Skew is:", df.lower.skew())

plt.hist(df.lower, color='palegreen')

plt.show()
target = np.log(df.lower)

print ("Skew is:", target.skew())

plt.hist(target, color='purple')

plt.show()
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as pp

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.express as px
df.plot(kind='scatter', x='lower', y='upper.1', alpha=0.5, color='darkgreen', figsize = (12,9))

plt.title('lower And upper.1')

plt.xlabel("lower")

plt.ylabel("upper.1")

plt.show()
trace1 = go.Box(

    y=df["lower"],

    name = 'lower',

    marker = dict(color = 'rgb(0,145,119)')

)

trace2 = go.Box(

    y=df["upper.1"],

    name = 'upper.1',

    marker = dict(color = 'rgb(255, 111, 145)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='year', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)

iplot(fig)
deathsIndSec=df.groupby(['lower','upper','median'])['upper.1'].mean().reset_index()



fig = px.scatter_mapbox(deathsIndSec[deathsIndSec["lower"]=='Nan'], 

                        lat="upper", lon="median",size="lower",size_max=12,

                        color="lower", color_continuous_scale=px.colors.sequential.Inferno, zoom=11)

fig.update_layout(mapbox_style="stamen-terrain")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
ax = sns.swarmplot(x="lower", y="upper", data=df)
f,ax = plt.subplots(figsize=(8,6))

sns.heatmap(df.corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
df.plot(kind='scatter', x='lower', y='upper.1', alpha=0.5, color='darkblue', figsize = (12,9))

plt.title('lower And upper.1')

plt.xlabel("lower")

plt.ylabel("upper.1")

plt.show()
f,ax=plt.subplots(1,2,figsize=(15,7))

df.lower.plot.hist(ax=ax[0],bins=30,edgecolor='black',color='crimson')

ax[0].set_title('lower')

x1=list(range(-150,350,50))

ax[0].set_xticks(x1)

plt.show()
labels1=df.lower.value_counts().index

sizes1=df.lower.value_counts().values

plt.figure(figsize=(11,11))

plt.pie(sizes1,labels=labels1,autopct="%1.1f%%")

plt.title("lower",size=25)

plt.show()