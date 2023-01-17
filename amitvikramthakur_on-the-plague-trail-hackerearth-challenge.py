import os # accessing directory structure

print(os.listdir('../input'))
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt



import sklearn

from sklearn import linear_model

from sklearn.metrics import mean_squared_error



import seaborn as sns

import missingno as msno

from tqdm import tqdm_notebook
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

df = pd.read_csv('../input/train.csv',encoding='latin1')

df.shape
df_trainset = df[0:int(0.7*df.shape[0])]

df_validationset = df[int(0.7*df.shape[0]):]
print(df_trainset.shape)

print(df_validationset.shape)
df_trainset.dataframeName = 'training dataset'

nRow, nCol = df_trainset.shape

print(f'There are {nRow} rows and {nCol} columns')
df_trainset.head()
# df_trainset.info() shows the number of non-null values (and hence the number of missing values).

df_trainset.info()
plotPerColumnDistribution(df_trainset, 10, 2)
plotCorrelationMatrix(df_trainset, 8)
plotScatterMatrix(df_trainset, 20, 10)
# missingno.matrix function shows missing/non-missing values in two colors.

msno.matrix(df_trainset)
# missingno.heatmap function gives nullity correlation

# (how strongly the presence or absence of one variable affects the presence of another).

# For the train dataset given, there is no strong correlation observed for nullity.

try:

    msno.heatmap(df_trainset)

except ValueError:

    pass
# Range of Date

#df_trainset['DateTime']
# Columns of Non-numeric Type

columns_total = np.array(list(df_trainset))

columns_numeric = np.array(list(df_trainset.select_dtypes(include=np.number)))

columns_non_numeric = np.setdiff1d(columns_total,columns_numeric)

print('---All Columns---')

print(columns_total)

print('---Columns of Numeric Type---')

print(columns_numeric)

print('---Columns of Non-numeric Type---')

print(columns_non_numeric)

# Data Visualizations

X_train = df_trainset[df_trainset.columns[0:30]]

y_train = df_trainset[df_trainset.columns[30:37]]

X_validate = df_validationset[df_validationset.columns[0:30]]

y_validate = df_validationset[df_validationset.columns[30:37]]
X_train.head()
y_train.head()
X_validate.head()
y_validate.head()
def function_preprocess_X(X):

    X_preprocessed = X.copy()

    X_preprocessed = X_preprocessed.select_dtypes(include=np.number)

    #X_train_numeric.head()

    return X_preprocessed
model_LinearRegression = sklearn.linear_model.LinearRegression()

X_train_preprocessed = function_preprocess_X(X_train)

model_LinearRegression.fit(X_train_preprocessed,y_train)
y_train_prediction = model_LinearRegression.predict(X_train_preprocessed)



rmse = np.sqrt(mean_squared_error(y_train,y_train_prediction))

print(f'RMSE: {rmse}')
X_validate_preprocessed = function_preprocess_X(X_validate)

y_validate_prediction = model_LinearRegression.predict(X_validate_preprocessed)



rmse = np.sqrt(mean_squared_error(y_validate,y_validate_prediction))

print(f'RMSE: {rmse}')
df_sample = pd.read_csv('../input/sample.csv',encoding='latin1')

print(df_sample.shape)

df_sample.head()
X_test = pd.read_csv('../input/test.csv',encoding='latin1')

print(X_test.shape)

X_test.head()
submission = pd.DataFrame(columns=df_sample.columns)

submission['ID'] = X_test['ID'].values

X_test_preprocessed = function_preprocess_X(X_test)
submission[y_train.columns] = model_LinearRegression.predict(X_test_preprocessed)

submission.to_csv('submission.csv',index=False)