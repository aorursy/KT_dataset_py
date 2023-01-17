import math
from scipy.io import arff
import pandas as pd
from pandas import Series
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from scipy.stats import expon
from scipy import stats
import seaborn as sns
plt.style.use('fivethirtyeight')
%matplotlib inline
df_fp = pd.read_excel('../input/fp.xlsx', delim_whitespace = True,
names=['id', 'Project', 'TeamExp', 'ManagerExp', 'YearEnd', 'Length', 'Effort', 'Transactions','Entities', 'PointsNonAdjust', 'Adjustment', 'PointsAjust','Language']
)
df_fp.dataframeName = 'fp.xlsx'
labels = list('AB')
df_fp.head(10)
plt.figure(figsize=(10,5))
plt.title('Effort cost estimation')
plt.grid(True)
plt.hist(df_fp.Effort, bins = 100, alpha=1, color = "skyblue",  linewidth=1)
plt.show()
plt.figure(figsize=(10,5))
plt.title('Entities')
plt.grid(True)
plt.hist(df_fp.Entities, bins = 100, alpha=1, color = "y",  linewidth=1)
plt.show()
plt.figure(figsize=(10,5))
plt.title('Transactions')
plt.grid(True)
plt.hist(df_fp.Transactions, bins = 100, alpha=1, color = "firebrick",  linewidth=1)
plt.show()
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
plotPerColumnDistribution(df_fp, 10, 3)
sns.distplot(df_fp.Effort);
print (df_fp.Effort.describe())
print (df_fp.Transactions.describe())
print (df_fp.TeamExp.describe())
print (df_fp.ManagerExp.describe())
plt.figure(figsize=(10,5))
plt.title('Languages')
plt.grid(True)
plt.hist(df_fp.Language, bins = 100, alpha=1,  linewidth=1, color=['orange'])
plt.show()
# Esfuerzo
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(df_fp.Effort)
for flier in bp['fliers']:
    flier.set(marker='o', color='red', alpha=0.5)

#Numero de transaciones
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(df_fp.Transactions)
for flier in bp['fliers']:
    flier.set(marker='o', color='red', alpha=0.5)
#Experiencia del equipo (A), experiencia del lider (B).
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
data_to_plot = [df_fp.TeamExp, df_fp.ManagerExp]
bp = ax.boxplot(data_to_plot, labels=labels,showmeans=True, meanline=True)
for flier in bp['fliers']:
    flier.set(marker='o', color='red', alpha=0.5)
#Puntos de Funcion no ajustados (A), Ajustados (B).
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
data_to_plot = [df_fp.PointsNonAdjust, df_fp.PointsAjust]
bp = ax.boxplot(data_to_plot, labels=labels,showmeans=True, meanline=True)
for flier in bp['fliers']:
    flier.set(marker='o', color='red', alpha=0.5)