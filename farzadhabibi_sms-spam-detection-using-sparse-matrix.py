import os

print(os.listdir("../input")) # For kaggle kernel
import pandas as pd 

import numpy as np

DATASET_DIR = '../input'  # For kaggle kernel

# DATASET_DIR = './datasets/spam'

DATASET_NAME = 'spam.csv'

dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)

dataset = pd.read_csv(dataset_path, encoding='ISO-8859-1')
dataset.head()
X = dataset.drop('v1', axis=1)

y = dataset['v1']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.head()
X_train.info()
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelection(BaseEstimator, TransformerMixin):

    def __init__(self, attrs):

        self.attrs = attrs

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return X[attrs]
attrs = ['v2']

selector = DataFrameSelection(attrs)

X = selector.transform(X_train)

X.head()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)

y_test = encoder.fit_transform(y_test)

y_train
Xy = X

Xy['status'] = y_train

Xy.head()
import matplotlib.pyplot as plt

plt.bar(('ham', 'spam'), (len(Xy[Xy['status'] == 0]),len(Xy[Xy['status'] == 1])), color='r')

prob = len(Xy[Xy['status'] == 1])/len(Xy)

print(f"prob of spam is {prob}")
Xy.groupby('status').describe()
class LengthAdder(BaseEstimator, TransformerMixin):

    def __init__(self, msg_attr='v2'):

        self.msg_attr = msg_attr

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        dataset = X.copy()

        dataset['len'] = X[self.msg_attr].apply(len)

        return dataset
Xy = LengthAdder().transform(Xy)

Xy.head()
Xy.hist(column='len', by='status', bins=25, figsize=(15, 7))
class ToLowerCase(BaseEstimator, TransformerMixin):

    def __init__(self, msg_attr='v2'):

        self.msg_attr = msg_attr

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        dataset = X.copy()

        dataset[self.msg_attr] = dataset[self.msg_attr].str.lower()

        return dataset

Xy = ToLowerCase().transform(Xy)

Xy.head()
from nltk.corpus import stopwords

from nltk.tokenize import WhitespaceTokenizer

import nltk 

class DeleteStopWordsAndPunc(BaseEstimator, TransformerMixin):

    def __init__(self, lang='english', msg_attr='v2'):

        self.lang = lang

        self.msg_attr = msg_attr

        nltk.download('stopwords')

        self.stop_words = set(stopwords.words('english')) 

        self.punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~+~'''

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        dataset = X.copy()

        dataset[self.msg_attr] = dataset[self.msg_attr].apply(self.delete_stop_words)

        dataset[self.msg_attr] = dataset[self.msg_attr].apply(self.delete_punc)

        return dataset

    def delete_stop_words(self, msg):

        tokens = WhitespaceTokenizer().tokenize(msg)

        return ' '.join([w for w in tokens if w not in self.stop_words])

    def delete_punc(self, msg):

        no_punct = ""

        for char in msg:

            if char not in self.punctuations:

                no_punct = no_punct + char

        return no_punct
Xy.head()
Xy = DeleteStopWordsAndPunc().transform(Xy)

Xy.head()
from nltk.stem import PorterStemmer

class Stemmer(BaseEstimator, TransformerMixin):

    def __init__(self, msg_attr='v2'):

        self.msg_attr = msg_attr

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        dataset = X.copy()

        dataset[self.msg_attr] = dataset[self.msg_attr].apply(self.stemming)

        return dataset

    def stemming(self, msg):

        return ' '.join([PorterStemmer().stem(w) for w in WhitespaceTokenizer().tokenize(msg)])
Xy = Stemmer().transform(Xy)

Xy.head()
import re #regex

class ChangeNumbersAndUrls(BaseEstimator, TransformerMixin):

    def __init__(self, msg_attr='v2'):

        self.msg_attr = msg_attr

        self.url_regex = re.compile(

                    r'^(?:http|ftp)s?://' # http:// or https://

                    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...

                    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip

                    r'(?::\d+)?' # optional port

                    r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        dataset = X.copy()

        dataset[self.msg_attr] = dataset[self.msg_attr].apply(self.change_urls_numbers)

        return dataset

    def change_urls_numbers(self, msg):

        tokens = WhitespaceTokenizer().tokenize(msg)

        new_msg = []

        for w in tokens:

            if w.isnumeric() :

                new_msg.append('NUMBER')

            elif re.match(self.url_regex, w):

                new_msg.append('URL')

            else : new_msg.append(w)

        return ' '.join(new_msg)
Xy = ChangeNumbersAndUrls().transform(Xy)

Xy.head()
class CreateExistenceMatrix(BaseEstimator, TransformerMixin):

    def __init__(self, msg_attr='v2', len_index=2):

        self.word_set = set({})

        self.msg_attr = msg_attr

        self.len_index = len_index

    def fit(self, X, y=None):

        for m in X[self.msg_attr]: 

            for w in WhitespaceTokenizer().tokenize(m):

                self.word_set.add(w)

        return self

    def transform(self, X, y=None):

        dataset = X.copy()

        for word in self.word_set:

            dataset[word] = 0

        indices = dataset.columns[self.len_index:]

        for index, row in dataset.iterrows():

            tokens = WhitespaceTokenizer().tokenize(row[self.msg_attr])

            for word in tokens:

                if word in dataset.columns:

                    dataset.at[index, word] = dataset.at[index, word] + 1

        return dataset

    def fit_transform(self, X, y=None):

        return self.fit(X, y).transform(X, y)
Xy = CreateExistenceMatrix(len_index=3).fit_transform(Xy)
from scipy.sparse import csr_matrix

class ToSparseMatrix(BaseEstimator, TransformerMixin):

    def __init__(self, msg_attr='v2'):

        self.msg_attr = msg_attr

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return csr_matrix(X.drop(self.msg_attr, axis=1).values)

Xy = ToSparseMatrix().transform(Xy)
from sklearn.pipeline import Pipeline

attrs = ['v2']

full_pipeline = Pipeline([

    ('selector', DataFrameSelection(attrs)),

    ('length_adder', LengthAdder()),

    ('to_lower', ToLowerCase()),

    ('delete_stop_punc', DeleteStopWordsAndPunc()),

    ('stemming', Stemmer()),

    ('change_number_url', ChangeNumbersAndUrls()),

    ('create_existance', CreateExistenceMatrix()),

    ('to_sparse', ToSparseMatrix()),

])
X_train_sparse = full_pipeline.fit_transform(X_train)
from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

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

from sklearn.metrics import f1_score, recall_score, precision_score, precision_recall_curve, roc_curve

scores = pd.DataFrame([], columns=['model_name', 'f1', 'precision', 'recall'])

for model in models:

    predictions = cross_val_predict(model[1], X_train_sparse, y_train, cv=3, n_jobs=-1)

    scores = scores.append({

        'model_name':model[0],

        'f1': f1_score(y_train, predictions),

        'precision': precision_score(y_train, predictions),

        'recall': recall_score(y_train, predictions),

    }, ignore_index=True)
scores
def show_plot_precision_recall(precision, recall, threshold, model_name, ax):

    ax.plot(threshold, precision[:-1], "b--", label="Precision")

    ax.plot(threshold, recall[:-1], "r-", label="Recall")

    ax.set_xlabel("Threshold")

    ax.set_ylabel(model_name)

    ax.legend(loc="lower right")

    ax.set_ylim([0, 1])
def draw_models_score_plot(models):

    i = 1

    fig = plt.figure(figsize=(20, 15))

    fig.subplots_adjust(hspace=0.4)

    for model in models:

        ax = fig.add_subplot(3, 3, i)

        i+= 1

        y_scores = []

        if hasattr(model[1], "decision_function"):

            y_scores = cross_val_predict(model[1], X_train_sparse, y_train, cv=3, method="decision_function", n_jobs=-1)

        else :

            y_probs = cross_val_predict(model[1], X_train_sparse, y_train, cv=3, method="predict_proba", n_jobs=-1)

            y_scores = y_probs[:, -1]

        precision, recall, thresholds = precision_recall_curve(y_train, y_scores)

        show_plot_precision_recall(precision, recall, thresholds, model[0], ax)

draw_models_score_plot(models) 
models = [

    ('mutlinomial_nb_0.2', MultinomialNB(alpha=0.2)),

    ('mutlinomial_nb_0.5', MultinomialNB(alpha=0.5)),

    ('mutlinomial_nb_1', MultinomialNB(alpha=1.0)),

    ('complement_nb_0.2', ComplementNB(alpha=0.2)),

    ('complement_nb_0.5', ComplementNB(alpha=0.5)),

    ('complement_nb_1', ComplementNB(alpha=1)),

    ('bernoli_nb_0.2', BernoulliNB(alpha=0.2)),

    ('bernoli_nb_0.5', BernoulliNB(alpha=0.5)),

    ('bernoli_nb_1', BernoulliNB(alpha=1)),

]

draw_models_score_plot(models)
from sklearn.model_selection import GridSearchCV

estimator = BernoulliNB()

grid_params = {

    'alpha': [0, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],

    'fit_prior': [True, False],

}

grid_search = GridSearchCV(estimator, grid_params, scoring='recall')

grid_search.fit(X_train_sparse, y_train)

grid_search.best_score_
best_estimator = grid_search.best_estimator_

best_estimator
y_probs = cross_val_predict(best_estimator, X_train_sparse, y_train, cv=3, method="predict_proba", n_jobs=-1)

y_scores = y_probs[:, -1]

precision, recall, thresholds = precision_recall_curve(y_train, y_scores)

ax = plt.axes()

show_plot_precision_recall(precision, recall, thresholds, "Bernoli NB", ax)
def predict(estimator, X, threshold=0.5):

    y_probs = estimator.predict_proba(X)[:,1]

    return (y_probs >= threshold)
X_test_transformed = full_pipeline.transform(X_test)
test_predictions = predict(best_estimator, X_test_transformed, 0.4)

recall = recall_score(y_test, test_predictions)

precision = precision_score(y_test, test_predictions)

f1 = f1_score(y_test, test_predictions)

print(f'Precision is {precision}\nRecall is {recall}\nF1 Score is {f1}')