from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
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

nRowsRead = None # specify 'None' if want to read whole file

# city_temperature.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/city_temperature.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'city_temperature.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
df1.isnull().sum()
df1.describe().round(1)
df1.boxplot(column='Year')
df1.boxplot(column='AvgTemperature')
print((df1['Year'] == 200).sum())

print((df1['AvgTemperature'] == -99).sum())
df1 = df1.drop(df1[(df1.Year == 200) | (df1.AvgTemperature == -99)].index)

df1 = df1.drop('State',1)

df1.head()
df1.groupby(['Year']).count()[['AvgTemperature']]
df1[(df1.Year != 2020)].groupby(['City','Year'])[['AvgTemperature']].count().describe()
df1.groupby('Region').describe()['AvgTemperature']
ax = sns.lineplot(x = 'Month', y = 'AvgTemperature', hue = 'Region', data = df1[(df1.Year != 2020)])

ax.set(xlabel = 'Month', ylabel='Average Temperature')

plt.show()
ax = sns.lineplot(x = 'Year', y = 'AvgTemperature', hue = 'Region', data = df1[(df1.Year != 2020)])

ax.set(xlabel = 'Year', ylabel='Average Temperature')

plt.show()
ax = sns.lineplot(x = 'Year', y = 'AvgTemperature', data = df1[(df1.Year != 2020)])

ax.set(xlabel = 'Year', ylabel='Average Temperature')

plt.show()
X = df1.copy()

y = X.AvgTemperature



X_train = X[(X.Year < 2015)].copy()

X_valid = X[(X.Year > 2014) & (X.Year != 2020)].copy()

X_test = X[(X.Year == 2020)].copy()



y_train = X_train.AvgTemperature

y_valid = X_valid.AvgTemperature

y_test = X_test.AvgTemperature



X_train.drop(['AvgTemperature'], axis=1, inplace=True)

X_valid.drop(['AvgTemperature'], axis=1, inplace=True)

X_test.drop(['AvgTemperature'], axis=1, inplace=True)
print(X_train.Year.unique())

print(X_valid.Year.unique())

print(X_test.Year.unique())
from sklearn.preprocessing import LabelEncoder



# Get list of categorical variables

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



label_X_train = X_train.copy()

label_X_valid = X_valid.copy()

label_X_test = X_test.copy()



label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    label_X_test[col] = label_encoder.transform(X_test[col])
label_X_train.head()
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



model = XGBRegressor(n_estimators=1000,learning_rate=0.05,n_jobs=-1)

model.fit(label_X_train,y_train, early_stopping_rounds=5, eval_set=[(label_X_valid,y_valid)])

predictions = model.predict(label_X_test)

mae = mean_absolute_error(predictions,y_test)

print("Mean Absolute Error:" , mae)
pd.DataFrame(predictions).head()
pd.DataFrame(y_test).head()