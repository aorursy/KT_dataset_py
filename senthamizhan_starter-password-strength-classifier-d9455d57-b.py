from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')
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

nRowsRead = 10000 # specify 'None' if want to read whole file

# data.csv has 669879 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/data.csv', error_bad_lines = False, nrows = nRowsRead)
df1.shape
plotPerColumnDistribution(df1, 10, 5)
import seaborn as sns

sns.countplot(df1['strength'])
df1.head()
df1['char_count'] = df1['password'].str.len()

df1['numerics'] = df1['password'].apply(lambda x: len([str(x) for x in list(x) if str(x).isdigit()]))

df1['alpha'] = df1['password'].apply(lambda x: len([x for x in list(x) if x.isalpha()]))



vowels = ['a', 'e', 'i', 'o', 'u']

df1['vowels'] = df1['password'].apply(lambda x: len([x for x in list(x) if x in vowels]))

df1['consonants'] = df1['password'].apply(lambda x: len([x for x in list(x) if x not in vowels and x.isalpha()]))



df1.head()
y = df1['strength']

X = df1.drop('strength', axis = 1)

X.head()
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from sklearn.preprocessing import StandardScaler



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=np.random)



svm = SVC(kernel="rbf", C=1.0, degree=3, coef0=1.0, random_state=np.random.RandomState())

X_train = X_train.drop('password', axis = 1)

scaler = StandardScaler().fit(X_train)

X_train_tr = scaler.transform(X_train)

svm.fit(X_train_tr, y_train)

X_test = X_test.drop('password', axis = 1)

X_test_tr = scaler.transform(X_test)

y_pred = svm.predict(X_test_tr)



print("Accuracy of SVM Classifier: {0:.2f}".format(balanced_accuracy_score(y_test, y_pred)))

scores = cross_val_score(svm, X_test_tr, y_test, cv = 10)

print("Mean accuracy after 10-fold cross validation: {0:.2f}".format(scores.mean()))
g = sns.pairplot(data = df1, hue = 'strength', vars = df1.columns.difference(['password', 'strength']), 

                 diag_kind = {'kde'}, palette = 'husl')