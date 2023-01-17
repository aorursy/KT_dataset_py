import warnings

warnings.filterwarnings("ignore")

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

# innercity.csv has 21613 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/innercity.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'innercity.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
df1.describe(include='all').T
df1.isna().sum()
df1.corr()
print(pd.isnull(df1).any())
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

x = df1.drop(columns=['price','dayhours'])

y = df1['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=9)

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

train_Pred = logreg.predict(x_train)


print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_squared_error



# user variables to tune

folds   = 10

metric  = "neg_mean_squared_error"



# hold different regression models in a single dictionary

models = {}

models["Linear"]        = LinearRegression()

models["Lasso"]         = Lasso()

models["ElasticNet"]    = ElasticNet()

models["KNN"]           = KNeighborsRegressor()

models["DecisionTree"]  = DecisionTreeRegressor()

models["SVR"]           = SVR()

models["AdaBoost"]      = AdaBoostRegressor()

models["GradientBoost"] = GradientBoostingRegressor()

models["RandomForest"]  = RandomForestRegressor()

models["ExtraTrees"]    = ExtraTreesRegressor()



# 10-fold cross validation for each model

model_results = []

model_names   = []

for model_name in models:

	model   = models[model_name]

	k_fold  = KFold(n_splits=folds, random_state=9)

	results = cross_val_score(model, x_train, y_train, cv=k_fold, scoring=metric)

	

	model_results.append(results)

	model_names.append(model_name)

	print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))



# box-whisker plot to compare regression models

figure = plt.figure()

figure.suptitle('Regression models comparison')

axis = figure.add_subplot(111)

plt.boxplot(model_results)

axis.set_xticklabels(model_names, rotation = 45, ha="right")

axis.set_ylabel("Mean Squared Error (MSE)")

plt.margins(0.05, 0.1)

plt.savefig("model_mse_scores.png")

plt.clf()

plt.close()