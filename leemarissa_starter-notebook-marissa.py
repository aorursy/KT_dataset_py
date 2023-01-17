# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
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
        ax[i, j].annotate('%.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file
# metatable_with_viral_status.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
meta = pd.read_csv('/kaggle/input/data/metatable_with_viral_status.csv', delimiter=',', nrows = nRowsRead, index_col=0)
meta.dataframeName = 'metatable_with_viral_status.csv'
nRow, nCol = meta.shape
print(f'There are {nRow} rows and {nCol} columns')
!head /kaggle/input/data/metatable_with_viral_status.csv
meta.head(5)
meta['viral_status']=='SC2'
meta[meta['viral_status']=='SC2']['age'].hist(alpha=0.5)
meta[meta['viral_status']=='no_virus']['age'].hist(alpha=0.5)
plt.show()
meta['sequencing_batch'].nunique()
meta['sequencing_batch'].unique()
meta['sequencing_batch'].value_counts()

meta['age'].describe()
meta['age'].hist()
plotPerColumnDistribution(meta, 10, 5)
plotCorrelationMatrix(meta, 8)
plotScatterMatrix(meta, 6, 15)
nRowsRead = 1000 # specify 'None' if want to read whole file
# swab_gene_counts.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
expr = pd.read_csv('/kaggle/input/data/swab_gene_counts.csv', delimiter=',', nrows = nRowsRead, index_col=0)
expr.dataframeName = 'swab_gene_counts.csv'
nRow, nCol = expr.shape
print(f'There are {nRow} rows and {nCol} columns')
def getGeneFrequency(status):
    rrs = meta[meta['viral_status']==status].index
    for i in range(len(rrs)):
        nums=expr.loc[:,[rrs[i]]].iloc[:,0]
        n=nums.idxmax()
        if n in maxes:
            maxes[n]+=1
        else:
            maxes[n]=1
    return maxes
dct=getGeneFrequency("no_virus")
df=pd.DataFrame(dct.values(), index=dct.keys(), columns=["frequency"])
df.plot.bar(legend=False)
plt.xlabel("Genes")
plt.ylabel("Frequency")
plt.title("Frequency of most common genes in patients with SC2")
plt.show()
expr.head(100)

names = pd.read_csv('/kaggle/input/annotation/gene2name.txt', delimiter='\t', nrows = nRowsRead, names=["Gene Name",'a'], index_col=0)
names.dataframeName = 'gene2name.txt'
names = names.drop(columns=["a"])
names.head()
for name in names.index:
    exprR=expr.rename(index={name:names.loc[name, "Gene Name"]})
exprR.head()

# n="ENSG00000000938"
# names.loc[n, "Gene Name"]
# exprR=expr.rename(index={n:names.loc[n, "Gene Name"]})
# exprR.head()
# transpose and reorder
exprT = expr.T.loc[meta.index,:]
exprT.dataframeName = 'swab_gene_counts.csv'
exprT.head(5)

plotCorrelationMatrix(expr, 24)
plotScatterMatrix(expr, 20, 10)