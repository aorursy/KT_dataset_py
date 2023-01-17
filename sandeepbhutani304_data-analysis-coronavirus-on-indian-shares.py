from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

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

df = pd.read_csv('/kaggle/input/effect-of-coronavirus-on-indian-stock-market/2020-03-30-NSE-EQ.txt')

df["date_1"]=pd.to_datetime(df["<date>"], format='%Y%m%d', errors='ignore')

print(df.columns)

print("Sample data  ----------------")

print(df.head())

print("List of share codes  --------------")

for ticker in df['<ticker>'].unique():

    print(ticker)

dfs = []

for dirname, _, filenames in os.walk('/kaggle/input/effect-of-coronavirus-on-indian-stock-market/'):

    for filename in filenames:

        dfs.append(pd.read_csv(os.path.join(dirname, filename)))



df_all=pd.concat(dfs, axis=0)

df_all["date_1"]=pd.to_datetime(df_all["<date>"], format='%Y%m%d', errors='ignore')

print("Total records: ", len(df_all))
# set draw_lockdown_lines = None if don't want to plot lockdates dates markers, Can also pass list of selected dates for markers

def plot_share_history(dfplot, tickername, draw_lockdown_lines=["2020-03-25", "2020-04-15","2020-05-04","2020-05-18"]):

    dfplot=df_all[["date_1","<close>"]][df_all["<ticker>"] == tickername]

    dfplot=dfplot.set_index("date_1")

    ax=dfplot.plot(title="Share Price history - {}".format(tickername))

    if draw_lockdown_lines is not None:

        for lock_date in draw_lockdown_lines:

            ax.axvline(lock_date, color="red", linestyle="--")

plot_share_history(df_all, "AUBANK")

plot_share_history(df_all, "SBIN")

plot_share_history(df_all, "AARTIIND")

plot_share_history(df_all, "CIPLA")

plot_share_history(df_all, "LT")
