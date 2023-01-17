import os

print(os.listdir("../input"))
import pandas as pd

import os



def load_data(path, filename, codec='utf-8'):

  csv_path = os.path.join(path, filename)

  print(csv_path)

  return pd.read_csv(csv_path, encoding=codec)



spam = load_data('../input', 'spam.csv', codec='latin1')
spam.head()
spam.columns = ['label', 'line1', 'line2', 'line3', 'line4']

spam.describe()
import seaborn as sns

import matplotlib.pyplot as plt



f, axs = plt.subplots(1, 2, figsize=(12, 6))

sns.countplot(spam['label'], ax=axs[0])

axs[1].pie(spam.groupby(spam['label'])['line1'].count(), labels=['ham', 'spam'], autopct='%1.1f%%', startangle=90, pctdistance=0.85)

plt.show()
spam_with_more_line = spam[spam['line2'].notnull()]

f, axs = plt.subplots(1, 2, figsize=(12, 6))

sns.countplot(spam_with_more_line['label'], ax=axs[0])

axs[1].pie(spam_with_more_line.groupby(spam_with_more_line['label'])['line1'].count(), labels=['ham', 'spam'], autopct='%1.1f%%',

           startangle=90, pctdistance=0.85)

plt.show()
spam['line2'].fillna('', inplace=True)

spam['line3'].fillna('', inplace=True)

spam['line4'].fillna('', inplace=True)



spam['text'] = spam['line1'] + ' ' + spam['line2'] + ' ' + spam['line3'] + ' ' + spam['line4']



spam.drop(['line1', 'line2', 'line3', 'line4'], axis=1, inplace=True)



spam.head()
spam_with_len = spam.copy()

spam_with_len['len'] = spam['text'].str.len()



spam_with_len.hist(column='len', by='label', bins=25, figsize=(15, 6), color = "skyblue")

plt.show()
spam = load_data('../input', 'spam.csv', 'latin1')

txts = spam.drop(['v1'], axis=1)

labels = spam['v1']

x_train, x_test, y_train, y_test = txts[:4457], txts[4457:], labels[:4457], labels[4457:]
label_map_func = lambda x: 1 if x == 'spam' else 0



y_test = list(map(label_map_func, y_test))

y_train = list(map(label_map_func, y_train))
x_test = x_test.reset_index().drop(['index'], axis=1)
import nltk

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk.tokenize.casual import TweetTokenizer

from nltk.stem import PorterStemmer 
from sklearn.base import BaseEstimator, TransformerMixin



class ConcatLines(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        pass

    def transform(self, X):

        X['Unnamed: 2'].fillna('', inplace=True)

        X['Unnamed: 3'].fillna('', inplace=True)

        X['Unnamed: 4'].fillna('', inplace=True)



        X['text'] = X['v2'] + ' ' + X['Unnamed: 2'] + ' ' + X['Unnamed: 3'] + ' ' + X['Unnamed: 4']

        X.drop(['v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

        return X

spam = ConcatLines().transform(spam)

spam.head()
class AddLength(BaseEstimator, TransformerMixin):

    def __init__(self, textAttr='text', lenAttr='len'):

        self.lenAttr = lenAttr

        self.textAttr = textAttr

    def fit(self, X, y=None):

        pass

    def transform(self, X):

        X[self.lenAttr] = X[self.textAttr].str.len()

        return X

spam = AddLength().transform(spam)

spam.head()
class ToLowerCase(BaseEstimator, TransformerMixin):

    def __init__(self, textAttr='text', lenAttr='len'):

        self.lenAttr = lenAttr

        self.textAttr = textAttr

    def fit(self, X, y=None):

        pass

    def transform(self, X):

        X[self.textAttr] = X[self.textAttr].str.lower()

        return X

    

spam = ToLowerCase().transform(spam)

spam.head()
class Tokenize(BaseEstimator, TransformerMixin):

    def __init__(self, textAttr='text', lenAttr='len'):

        self.lenAttr = lenAttr

        self.textAttr = textAttr

    def fit(self, X, y=None):

        pass

    def transform(self, X):

        x_len = X[self.lenAttr]

        x_text = X[self.textAttr]

        x_text = [TweetTokenizer().tokenize(str(x)) for x in x_text]

        X['text'] = x_text

        X['len'] = x_len

        return X

    

spam = Tokenize().transform(spam)

spam.head()
import string



class RemoveStopWordsAndStem:

    def __init__(self, textAttr='text', lenAttr='len'):

        self.lenAttr = lenAttr

        self.textAttr = textAttr

        self.ps = ps = PorterStemmer()

    def fit(self, X, y=None):

        pass

    def transform(self, X):

        alphabet = list(string.ascii_lowercase)

        stop_words = list(stopwords.words('english'))

        puncs = list(string.punctuation)

        stop_words = stop_words + puncs + alphabet

        x_text = X[self.textAttr]

        text = []

        for i in range(len(X)):

            filtered_sentence = []

            for w in x_text[i]: 

                if w not in stop_words:

                    w = w.rstrip(".")

                    if w is not "":

                        filtered_sentence.append(self.ps.stem(w))

            text.append(filtered_sentence)

        X[self.textAttr] = text

        return X

spam = RemoveStopWordsAndStem().transform(spam)

spam.head()
class Substitute:

    def __init__(self, textAttr='text', lenAttr='len'):

        self.lenAttr = lenAttr

        self.textAttr = textAttr

        self.emoji_list = emoji_list = [':', ';', '>', '=']

        self.website_list = ['.com', '.org', '.co.uk', '.net', 'http', 'www.']

    def fit(self, X, y=None):

        pass

    def transform(self, X):

        x_text = X[self.textAttr]

        text = []

        for i in range(len(X)):

            text.append(self.substitute(x_text[i]))

        X[self.textAttr] = text

        return X

    

    def substitute(self, words):

        for i in range(len(words)):

            if self.is_emoji(words[i]):

                words[i] = 'emoji'

            elif words[i].isnumeric():

                words[i] = 'digitnumber'

            else :

                for site in self.website_list:

                    if site in words[i]:

                        words[i] = 'website'

        return words

    def is_emoji(self, word):

        return word[0] in self.emoji_list and len(word) > 1



spam = Substitute().transform(spam)

spam.head()
from scipy import sparse

import numpy as np



class ToSparseMatrix:

    def __init__(self, train_set, textAttr='text', lenAttr='len'):

        self.lenAttr = lenAttr

        self.textAttr = textAttr

        self.train_words = tokenize_pipeline.transform(train_set.copy())

    def fit(self, X, y=None):

        pass

    def transform(self, X):

        if self.train_words is None:

            self.train_words = X

        self.final_words = np.array([x for t in self.train_words[self.textAttr] for x in t])

        self.final_words = np.unique(self.final_words)

        matrix = np.array([[0 for x in range(len(self.final_words) + 1)] for y in range(len(X))])

        x_texts = list(X[self.textAttr])

        x_len = list(X[self.lenAttr])

        for i in range(len(x_texts)):

            for token in x_texts[i]:

                cond = np.where(self.final_words == token)

                if(len(cond[0]) > 0):

                    matrix[i][cond[0][0]] += 1

            matrix[i][-1] = x_len[i]

        return sparse.csr_matrix(matrix)
from sklearn.pipeline import Pipeline



tokenize_pipeline = Pipeline([

    ('concat lines', ConcatLines()),

    ('add length', AddLength()),

    ('lower case words', ToLowerCase()),

    ('tokenize words', Tokenize()),

    ('remove stopwords', RemoveStopWordsAndStem()),

    ('substitute emoji, web, number', Substitute()),

])



sparse_pipeline = Pipeline([

    ('sparse pipeline', ToSparseMatrix(x_train.copy(deep=True)))

])



x_train_prepared = tokenize_pipeline.transform(x_train.copy(deep=True))

x_train_prepared = sparse_pipeline.transform(x_train_prepared)

x_train_prepared
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

models = [

    ('svc', SVC(kernel='rbf')),

    ('neighbors', KNeighborsClassifier(3)),

    ('random_forest', RandomForestClassifier()),

    ('sgd', SGDClassifier()), 

    ('mutlinomial_nb', MultinomialNB()),

    ('complement_nb', ComplementNB()),

    ('bernoli_nb', BernoulliNB()),

]
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score

scores = pd.DataFrame([], columns=['model', 'accuracy', 'auc', 'precision', 'recall', 'f1'])

for model in models:

    pred = cross_val_predict(model[1], x_train_prepared, y_train, cv=3, n_jobs=-1)

    scores = scores.append({

        'model':model[0],

        'accuracy' : accuracy_score(y_train, pred),

        'auc' : roc_auc_score(y_train, pred),

        'precision': precision_score(y_train, pred),

        'recall': recall_score(y_train, pred),

        'f1': f1_score(y_train, pred),

    }, ignore_index=True)

scores
from sklearn.model_selection import GridSearchCV



estimator = BernoulliNB()

grid_params = {

    'alpha': [0, 0.08, 0.09, 0.10, 0.11, 0.15],

    'binarize': [0, 0.1, 0.3, 0.5],

    'fit_prior': [True, False],

    'class_prior': [None, [0.4, 0.6]],

}

grid_search = GridSearchCV(estimator, grid_params, scoring='recall')

grid_search.fit(x_train_prepared, y_train)

grid_search.best_score_
final_model = grid_search.best_estimator_

final_model
x_test_prepared = tokenize_pipeline.transform(x_test.copy(deep=True))

x_test_prepared = sparse_pipeline.transform(x_test_prepared)

pred = cross_val_predict(final_model, x_test_prepared, y_test, cv=3)



print("Precision: ", precision_score(y_test, pred))

print("Recall: ", recall_score(y_test, pred))

print("f1_score: ", f1_score(y_test, pred))

print("Accuracy: ", accuracy_score(y_test, pred))