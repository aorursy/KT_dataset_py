from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm_notebook as tqdm
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
nRowsRead = 1000 # specify 'None' if want to read whole file

# animeListGenres.csv has 15729 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/animeListGenres.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'animeListGenres.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
nRowsRead = 1000 # specify 'None' if want to read whole file

# animeReviewsOrderByTime.csv has 135201 rows in reality, but we are only loading/previewing the first 1000 rows





with open('../input/animeReviewsOrderByTime.csv', 'r', encoding='utf-8') as f:

    headers = f.readline().replace('"','').replace('\n','').split(',')

    print(headers)

    print('The number of column: ', len(headers))

    dataFormat = dict()

    for header in headers:

        dataFormat[header] = list()



    for idx, line in enumerate(tqdm(f.readlines(), desc='Now parsing... ')):

        

        if idx == 67:

            yee = line

        

        if line != '':

            line = line.replace('\n','')

            indices = [i for i, x in enumerate(line) if x == ',']

            idxStart = 0

            for i in range(len(headers)):

                if i < len(headers) - 1:

                    dataFormat[headers[i]].append(line[idxStart + 1:indices[i] - 1])

                    idxStart = indices[i] + 1

                elif i == len(headers) - 1:

                    dataFormat[headers[i]].append(line[idxStart + 1:-1])

                else:

                    break

        if nRowsRead is not None and nRowsRead == idx + 1:

            print('We read only', nRowsRead, 'lines.')

            break
df2 = pd.DataFrame(dataFormat)

nRow, nCol = df2.shape

print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
print(df1.columns)

print(df2.columns)
%matplotlib inline

import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer

import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

from nltk.stem.porter import PorterStemmer

import string

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec, KeyedVectors

import pickle

import warnings

warnings.filterwarnings("ignore")

from sklearn import datasets, neighbors

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate

from collections import Counter

from matplotlib.colors import ListedColormap

import scikitplot.metrics as sciplot

from sklearn.metrics import accuracy_score

import math

import nltk

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
df2.head()
df2.describe()
plt.figure(figsize=(8,8))

ax=sns.countplot(df2['overallRating'],color='skyblue')

ax.set_xlabel("Score")

ax.set_ylabel('Count')

ax.set_title("Distribution of Review Score")
# Cleaning the texts

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, 10000):

    review = re.sub('[^a-zA-Z]', ' ', df2['review'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
corpus=pd.DataFrame(corpus, columns=['Reviews']) 

corpus.head()
result=corpus.join(df2[['overallRating']])

result.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

tfidf.fit(result['Reviews'])
X = tfidf.transform(result['Reviews'])

result['Reviews'][1]
print([X[1, tfidf.vocabulary_['show']]])
result['overallRating']=pd.to_numeric(result['overallRating'])

result.dropna(inplace=True)

result[result['overallRating'] != 3]

result['Positivity'] = np.where(result['overallRating'] > 3, 1, 0)

cols = [ 'overallRating']

result.drop(cols, axis=1, inplace=True)

result.head()
result.groupby('Positivity').size()
from sklearn.model_selection import train_test_split

X = result.Reviews

y = result.Positivity

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),

                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,

                                                                            (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))
print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_test),

                                                                             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,

                                                                            (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):

    sentiment_fit = pipeline.fit(X_train, y_train)

    y_pred = sentiment_fit.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("accuracy score: {0:.2f}%".format(accuracy*100))

    return accuracy
cv = CountVectorizer()

rf = RandomForestClassifier(class_weight="balanced")

n_features = np.arange(10000,25001,5000)



def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):

    result = []

    print(classifier)

    print("\n")

    for n in n_features:

        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)

        checker_pipeline = Pipeline([

            ('vectorizer', vectorizer),

            ('classifier', classifier)

        ])

        print("Test result for {} features".format(n))

        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)

        result.append((n,nfeature_accuracy))

    return result
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
print("Result for trigram with stop words (Tfidf)\n")

feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))
from sklearn.metrics import classification_report



cv = CountVectorizer(max_features=30000,ngram_range=(1, 3))

pipeline = Pipeline([

        ('vectorizer', cv),

        ('classifier', rf)

    ])

sentiment_fit = pipeline.fit(X_train, y_train)

y_pred = sentiment_fit.predict(X_test)



print(classification_report(y_test, y_pred, target_names=['negative','positive']))
## K-fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = pipeline, X= X_train, y = y_train,

                             cv = 10)

print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))
from sklearn.feature_selection import chi2



tfidf = TfidfVectorizer(max_features=30000,ngram_range=(1, 3))

X_tfidf = tfidf.fit_transform(result.Reviews)

y = result.Positivity

chi2score = chi2(X_tfidf, y)[0]
plt.figure(figsize=(16,8))

scores = list(zip(tfidf.get_feature_names(), chi2score))

chi2 = sorted(scores, key=lambda x:x[1])

topchi2 = list(zip(*chi2[-20:]))

x = range(len(topchi2[1]))

labels = topchi2[0]

plt.barh(x,topchi2[1], align='center', alpha=0.5)

plt.plot(topchi2[1], x, '-o', markersize=5, alpha=0.8)

plt.yticks(x, labels)

plt.xlabel('$\chi^2$')

plt.show();
from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re
max_fatures = 30000

tokenizer = Tokenizer(nb_words=max_fatures, split=' ')

tokenizer.fit_on_texts(result['Reviews'].values)

X1 = tokenizer.texts_to_sequences(result['Reviews'].values)

X1 = pad_sequences(X1)
Y1 = pd.get_dummies(result['Positivity']).values

X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1,Y1, random_state = 42)

print(X1_train.shape,Y1_train.shape)

print(X1_test.shape,Y1_test.shape)
embed_dim = 150

lstm_out = 200



model = Sequential()

model.add(Embedding(max_fatures, embed_dim,input_length = X1.shape[1], dropout=0.2))

model.add(LSTM(lstm_out, dropout_U=0.2,dropout_W=0.2))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
batch_size = 32

model.fit(X1_train, Y1_train, nb_epoch = 7, batch_size=batch_size, verbose = 2)
score,acc = model.evaluate(X1_test, Y1_test, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))
pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(X1_test)):

    

    result = model.predict(X1_test[x].reshape(1,X1_test.shape[1]),batch_size=1,verbose = 2)[0]

   

    if np.argmax(result) == np.argmax(Y1_test[x]):

        if np.argmax(Y1_test[x]) == 0:

            neg_correct += 1

        else:

            pos_correct += 1

       

    if np.argmax(Y1_test[x]) == 0:

        neg_cnt += 1

    else:

        pos_cnt += 1







print("pos_acc", pos_correct/pos_cnt*100, "%")

print("neg_acc", neg_correct/neg_cnt*100, "%")