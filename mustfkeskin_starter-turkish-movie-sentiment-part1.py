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

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = None # specify 'None' if want to read whole file



df = pd.read_csv('/kaggle/input/turkish_movie_sentiment_dataset.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'turkish_movie_sentiment_dataset.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(5)
plotPerColumnDistribution(df, 10, 5)
import gensim, logging

import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

 

intab    = 'ÇĞIİÖŞÜ'

outtab   = 'çğıiöşü'

tr_lower = str.maketrans(intab, outtab)



# Lower all character and remove number and punctiations

df["comment"] = df["comment"].apply(lambda x : x.translate(tr_lower).lower())

df["comment"] = df["comment"].apply(lambda x : re.sub(r"[^a-zçğıöşü]", ' ', x))

df["comment"] = df["comment"].apply(lambda x : re.sub(r"\s+", ' ', x))





# For word2vec training we need list of list format

# sentences = [['first', 'sentence'], ['second', 'sentence']]

sentences = df["comment"].apply(lambda x : x.split())
# train word2vec on the sentences

model = gensim.models.Word2Vec(sentences, min_count=3,  window=5, workers=4, size=300)
model.wv.most_similar("iyi")
model.wv.most_similar("kötü")
import numpy as np

df["point"] = df["point"].astype(str).str.replace(",", ".")

df["point"] = df["point"].astype(float)

df["label"] = np.where(df["point"] > 2.5, 1, 0)

df.head()
def get_mean_vector(word2vec_model, words):

    # remove out-of-vocabulary words

    words = [word for word in words if word in word2vec_model.wv]

    if len(words) >= 1:

        return np.mean(word2vec_model[words], axis=0)

    else:

        return np.zeros((1, word2vec_model.vector_size))
vectors = []

for sentence in sentences:

    vec = get_mean_vector(model, sentence)

    vectors.append(vec)
from sklearn.model_selection import train_test_split



vectors = np.array(vectors)

X = np.vstack(vectors)

y = df["label"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

rf.fit(X_train, y_train)
train_predictions = rf.predict(X_train)

test_predictions = rf.predict(X_test)

valid_predictions = rf.predict(X_valid)
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score



print("accuracy: " + str(accuracy_score(y_train, train_predictions)))

precision_recall_fscore_support(y_train, train_predictions)
print("accuracy: " + str(accuracy_score(y_test, test_predictions)))

precision_recall_fscore_support(y_test, test_predictions)
print("accuracy: " + str(accuracy_score(y_valid, valid_predictions)))

precision_recall_fscore_support(y_valid, valid_predictions)