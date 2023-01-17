%matplotlib inline

import numpy as np

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, roc_auc_score

import seaborn as sns

import re

import os

# Code heavily inspired from the following posts

# http://www.davidsbatista.net/blog/2017/04/01/document_classification/

# https://towardsdatascience.com/multi-label-text-classification-with-scikit-learn-30714b7819c5
# Load NLTK's English stop-words list

stop_words = set(stopwords.words('english'))

data_df = pd.read_csv("../input/sdgs.csv", delimiter=',')

data_df.dropna(inplace=True)

data_df.info()
df_sdgs = data_df.drop(['text'], axis=1)

counts = []

categories = list(df_sdgs.columns.values)

for i in categories:

    counts.append((i, df_sdgs[i].sum()))

df_stats = pd.DataFrame(counts, columns=['sdg', 'No. texts'])

#df_stats = pd.DataFrame(counts, columns=['sdg', '#texts']).to_latex(column_format='lr')

df_stats
df_stats.plot(x='sdg', y='No. texts', kind='bar', legend=False, grid=True, figsize=(10, 6))

plt.title("Number of documents per SDG")
rowsums = data_df.iloc[:,1:].sum(axis=1)

x= rowsums.value_counts()

plt.figure(figsize=(10,6))

ax = sns.barplot(x.index, x.values)

plt.title("SDG labels per document")

plt.ylabel('Number of documents', fontsize=12)

plt.xlabel('Number of labels', fontsize=12)
# Distribution by string length of documents

lens = data_df.text.str.len()

lens.hist(bins = np.arange(0,5000,100))
def clean_text(text):

    text = text.lower()

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip(' ')

    return text
# split the data, leave 1/3 out for testing

data_x = data_df[['text']].values

data_y = data_df.drop(['text'], axis=1).values



stratified_split = ShuffleSplit(n_splits=2, test_size=0.33)    

for train_index, test_index in stratified_split.split(data_x, data_y):

    x_train, x_test = data_x[train_index], data_x[test_index]

    y_train, y_test = data_y[train_index], data_y[test_index]





#transform matrix of plots into lists to pass to a TfidfVectorizer

train_x = [x[0] for x in x_train.tolist()]

test_x = [x[0] for x in x_test.tolist()]



train_y = [y[0] for y in y_train.tolist()]

test_y = [y[0] for y in y_test.tolist()]
labels = list(data_df.drop(['text'], axis=1).columns.values)



def grid_search(train_x, train_y, test_x, test_y, labels, parameters, pipeline):

    '''Train pipeline, test and print results'''

    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)

    grid_search_tune.fit(train_x, train_y)



    print()

    print("Best parameters set:")

    print(grid_search_tune.best_estimator_.steps)

    print()



    # measuring performance on test set

    print("Applying best classifier on test data:")

    best_clf = grid_search_tune.best_estimator_

    predictions = best_clf.predict(test_x)



    print(classification_report(test_y, predictions, target_names=labels))

    print("ROC-AUC:", roc_auc_score(test_y, predictions))
pipeline = Pipeline([

                ('tfidf', TfidfVectorizer(stop_words=stop_words)),

                ('clf', OneVsRestClassifier(MultinomialNB(

                    fit_prior=True, class_prior=None))),

            ])

parameters = {

                'tfidf__max_df': (0.25, 0.5, 0.75),

                'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],

                'clf__estimator__alpha': (1e-2, 1e-3)

            }

grid_search(train_x, y_train, test_x, y_test, labels, parameters, pipeline)
pipeline = Pipeline([

    ('tfidf', TfidfVectorizer(stop_words=stop_words)),

    ('clf', OneVsRestClassifier(LinearSVC())),

])

parameters = {

    'tfidf__max_df': (0.25, 0.5, 0.75),

    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],

    "clf__estimator__C": [0.01, 0.1, 1],

    "clf__estimator__class_weight": ['balanced', None],

}

grid_search(train_x, y_train, test_x, y_test, labels, parameters, pipeline)
pipeline = Pipeline([

    ('tfidf', TfidfVectorizer(stop_words=stop_words)),

    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'))),

])

parameters = {

    'tfidf__max_df': (0.25, 0.5, 0.75),

    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],

    "clf__estimator__C": [0.01, 0.1, 1],

    "clf__estimator__class_weight": ['balanced', None],

}

grid_search(train_x, y_train, test_x, y_test, labels, parameters, pipeline)