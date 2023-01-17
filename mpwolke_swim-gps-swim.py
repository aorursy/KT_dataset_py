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

df1 = pd.read_csv('../input/cusersmarildownloadsswimmingcsv/swimming.csv', delimiter=';', nrows = nRowsRead)

df1.dataframeName = 'swimming.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.tail(5)
df1.shape
df1.info
df1.describe()
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 15, 10)
df1.calories.describe()
print ("Skew is:", df1.calories.skew())

plt.hist(df1.calories, color='magenta')

plt.show()
target = np.log(df1.calories)

print ("Skew is:", target.skew())

plt.hist(target, color='lightgreen')

plt.show()
numeric_features = df1.select_dtypes(include=[np.number])

numeric_features.dtypes
corr = numeric_features.corr()



print (corr['calories'].sort_values(ascending=False)[1:11], '\n')

print (corr['calories'].sort_values(ascending=False)[-10:])
df1.calories.unique()
#Define a function which can pivot and plot the intended aggregate function 

def pivotandplot(data,variable,onVariable,aggfunc):

    pivot_var = data.pivot_table(index=variable,

                                  values=onVariable, aggfunc=aggfunc)

    pivot_var.plot(kind='bar', color='teal')

    plt.xlabel(variable)

    plt.ylabel(onVariable)

    plt.xticks(rotation=0)

    plt.show()
pivotandplot(df1,'calories','number_of_runs',np.median)
# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
# It is a continous variable and hence lets look at the relationship of calories with number_of_runs using a Regression plot



_ = sns.regplot(df1['calories'], df1['number_of_runs'])
df1.plot(kind='scatter', x='calories', y='number_of_runs', alpha=0.5, color='orangered', figsize = (12,9))

plt.title('calories And number_of_runs')

plt.xlabel("calories")

plt.ylabel("number_of_runs")

plt.show()
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling as pp

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.express as px
trace1 = go.Box(

    y=df1["calories"],

    name = 'calories',

    marker = dict(color = 'rgb(0,63,132)')

)

trace2 = go.Box(

    y=df1["number_of_runs"],

    name = 'number_of_runs',

    marker = dict(color = 'rgb(88, 171, 98)')

)



data = [trace1, trace2]

layout = dict(autosize=False, width=700,height=500, title='calories', paper_bgcolor='rgb(243, 243, 243)', 

              plot_bgcolor='rgb(243,243,243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)

iplot(fig)
ax = sns.scatterplot(x="distance", y="time", \

                     hue="favorite", legend="full", palette = "BuGn_r", data=df1)
ax = sns.violinplot(x="total_strokes", y="number_of_runs", data=df1, 

                    inner=None, color=".8")

ax = sns.stripplot(x="total_strokes", y="number_of_runs", data=df1, 

                   jitter=True)

ax.set_title('total_strokes vs number_of_runs')

ax.set_ylabel('Fast and furious')