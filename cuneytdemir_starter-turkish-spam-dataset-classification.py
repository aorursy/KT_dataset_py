from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

import nltk

from nltk.stem import PorterStemmer

import string

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

data=pd.read_csv('../input/trspam.csv',error_bad_lines=False)

df_docs = pd.DataFrame({'email': data.iloc[:,1].fillna(""), 'class': data.iloc[:,0].fillna("")})



nRow, nCol = df_docs.shape

print(f'There are {nRow} rows and {nCol} columns')
df_docs.head(5)
plotPerColumnDistribution(df_docs, 10, 5)
speeches = list(df_docs['email'])

labels = list(df_docs['class'])



training_size = int(len(speeches) * 0.8)

train_speeches = speeches[:training_size]

train_labels = labels[:training_size]

test_speeches = speeches[training_size:]

test_labels = labels[training_size:]



stemmer = PorterStemmer()

def tokenize_and_stem(text):

    tokens = nltk.tokenize.word_tokenize(text)

    # strip out punctuation and make lowercase

    tokens = [token.lower().strip(string.punctuation)

              for token in tokens if token.isalnum()]

    tokens = [stemmer.stem(token) for token in tokens]

    return tokens



vectorizer2 = CountVectorizer(tokenizer=tokenize_and_stem)

train_features_tokenized = vectorizer2.fit_transform(train_speeches)

test_features_tokenized = vectorizer2.transform(test_speeches)
from sklearn.svm import SVC



def train_classifier(train_speeches, train_labels, test_speeches, test_labels, classifier):

    train_features = CountVectorizer(tokenizer=tokenize_and_stem).fit_transform(train_speeches)

    classifier.fit(train_features_tokenized, train_labels)

    return classifier
classifier = train_classifier(train_speeches, train_labels, test_speeches, test_labels, SVC(kernel='linear'))

print (classifier.score(test_features_tokenized, test_labels))



classifier = train_classifier(train_speeches, train_labels, test_speeches, test_labels, MultinomialNB())

print(classifier.score(test_features_tokenized, test_labels))