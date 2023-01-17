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
df1 = pd.read_csv('../input/Consumo_cerveja.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Consumo_cerveja.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
import seaborn as sns
%matplotlib inline
df1.head()
df1.isnull().any()
df1.info()
df1
# major rows of data have NaN values, thus dropping them would be a wiser descision
df1.dropna(inplace=True)
# this reduces our dataset to great amount
df1.info()
df1['Temperatura Media (C)'] = df1['Temperatura Media (C)'].apply(lambda x : float(x.replace(',','.')))
df1['Temperatura Maxima (C)'] = df1['Temperatura Maxima (C)'].apply(lambda x : float(x.replace(',','.')))
df1['Temperatura Minima (C)'] = df1['Temperatura Minima (C)'].apply(lambda x : float(x.replace(',','.')))
df1['Precipitacao (mm)'] = df1['Precipitacao (mm)'].apply(lambda x : float(x.replace(',','.')))
df1.head()
df1.describe()
sns.pairplot(df1)
# the pairplot gives an idea that our label depends somewhat linearly with most of our features 
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(df1.drop(['Consumo de cerveja (litros)','Data'],axis=1),df1['Consumo de cerveja (litros)'])
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
from sklearn import metrics
metrics.r2_score(y_test,predictions)
# the score is not that much bad considering the dataset wasn't large enough
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))
plt.scatter(y_test,predictions)
p = sns.distplot(y_test,kde=False,bins=50)
p1 = sns.distplot(predictions,kde=False,bins=50)
p2 = sns.distplot(y_test-predictions,kde=False,bins=50)
lm.coef_
lm.intercept_
# attempt to try another regressor
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=50)
rfr.fit(X_train,y_train)
pred = rfr.predict(X_test)
rfr.score(X_test,y_test)
# Randomforestregressor didn't performed well as compared to linear regression which makes sense because our label depends linearly with many of 
# our features
