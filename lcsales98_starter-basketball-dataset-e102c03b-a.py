from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

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

# train.csv has 47525 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/train.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'train.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(10)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 10)
plotScatterMatrix(df1, 10, 10)
plt.figure(figsize=(12,10))

sns.barplot(x=' home_team_win', y=' visitor_team_name', data=df1)
plt.figure(figsize=(12,10))

sns.barplot(x=' home_team_win', y=' home_team_name', data=df1)
plt.figure(figsize=(30,10))

sns.barplot(x=' home_team_score', y=' visitor_team_score', data=df1)
one_hot_encoded_training_predictors = pd.get_dummies(df1)

one_hot_encoded_training_predictors.head(10)

len(one_hot_encoded_training_predictors.columns)
from sklearn.model_selection import train_test_split



new_train = one_hot_encoded_training_predictors

predictors = new_train.drop([' home_team_win'], axis=1)

target = df1[" home_team_win"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.25, random_state = 0)
from keras.models import Sequential

from keras.layers import Dense



regressor = Sequential()

#(64 + 1)/2 = 32

regressor.add(Dense(32, activation='relu', input_dim = 63))

regressor.add(Dense(32, activation='relu'))

regressor.add(Dense(units=1, activation='linear'))

regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error','acc'])

model = regressor.fit(x_train, y_train,batch_size=300, epochs=1000)
print(model.history.keys())
print(model.history.keys())

# summarize history for accuracy

plt.plot(model.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(model.history['loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predict = regressor.predict(x_val)



error = (sum(abs(a - b) for a, b in zip(predict, y_val))/ len(predict))[0]

print("O erro médio na predição é: {}".format(error))