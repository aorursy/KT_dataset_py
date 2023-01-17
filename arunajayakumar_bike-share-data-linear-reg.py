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

# bike_share.csv has 10886 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/bike_share.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'bike_share.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
df1.duplicated().sum()
df1[df1.duplicated()]
df1.info()
df1.corr()['count']
df1.drop(columns=['atemp','casual','registered'],inplace = True)
df1.head()
import seaborn as sns

plt.figure(figsize=(16, 6))

sns.boxplot(data=df1, orient="h")
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X = df1.drop(columns='count')

y = df1['count']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

model = LinearRegression()

model.fit(X_train,y_train)

print("coeffecient issssssssssss",model.coef_)

print("intercept is",model.intercept_)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
train_predict = model.predict(X_train)



mae_train = mean_absolute_error(y_train,train_predict)



mse_train = mean_squared_error(y_train,train_predict)



rmse_train = np.sqrt(mse_train)



r2_train = r2_score(y_train,train_predict)



mape_train = mean_absolute_percentage_error(y_train,train_predict)
test_predict = model.predict(X_test)



mae_test = mean_absolute_error(test_predict,y_test)



mse_test = mean_squared_error(test_predict,y_test)



rmse_test = np.sqrt(mean_squared_error(test_predict,y_test))



r2_test = r2_score(y_test,test_predict)



mape_test = mean_absolute_percentage_error(y_test,test_predict)
print('+****************+')

print('TRAIN: Mean Absolute Error(MAE): ',mae_train)

print('TRAIN: Mean Squared Error(MSE):',mse_train)

print('TRAIN: Root Mean Squared Error(RMSE):',rmse_train)

print('TRAIN: R square value:',r2_train)

print('TRAIN: Mean Absolute Percentage Error: ',mape_train)

print('+*************************+')

print('TEST: Mean Absolute Error(MAE): ',mae_test)

print('TEST: Mean Squared Error(MSE):',mse_test)

print('TEST: Root Mean Squared Error(RMSE):',rmse_test)

print('TEST: R square value:',r2_test)

print('TEST: Mean Absolute Percentage Error: ',mape_test)