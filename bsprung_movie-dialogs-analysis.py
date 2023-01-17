import numpy as np

import pandas as pd

pd.set_option('display.max_colwidth', -1)

import matplotlib.pyplot as plt

import re

import pickle

import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.dummy import DummyClassifier

from sklearn.feature_extraction.text import CountVectorizer

from nltk.tokenize import word_tokenize

import nltk

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)

# Gensim install and documentation:

#    https://radimrehurek.com/gensim/install.html

#    http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/

# Word2Vec tutorials:

#  Skip-gram model: similar to autoencoder, goal is to learn weights (word vectors) in an unsupervised way

#    The unsupervised task is, given a word, predict surrounding words

#    http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

#  Negative sampling: Same as before but make it feasible to train (300M weights reduced by

#    sub-sampling frequent words and using 'negative sampling' optimization)

#    http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

#  Applying Word2Vec to recommenders and advertising

#    http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/

# Word2Vec applications:

#    predict age/gender for a specific random blog

#    https://github.com/sunyam/Blog_Authorship
# Find the path to Chatbot Data files

# This search is useful because the spelling is slightly different in the Workspace

# For example 'Chatbot Data' in the Workspace translates into 'chatbot-data' in the path

import os

print("Files in '../input':")

for entry in os.scandir('../input'):

    print(entry.name)

print('')

print("Files in '../input/chatbot-data':")

for entry in os.scandir('../input/chatbot-data'):

    print(entry.name)

print('')

print("Files in '../input/chatbot-data/cornell_movie_dialogs_corpus':")

for entry in os.scandir('../input/chatbot-data/cornell_movie_dialogs_corpus'):

    print(entry.name)

print('')

print("Files in '../input/chatbot-data/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus':")

for entry in os.scandir('../input/chatbot-data/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus'):

    print(entry.name)
# Let's read the movie titles files and see whether we can find 'The Godfather'

titles = pd.read_csv('../input/chatbot-data/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus/movie_titles_metadata.txt'

                     , header = None, names = ['movieID', 'movie title', 'movie year', 'IMDB rating', 'IMDB votes', 'genres']

                     , sep = '\+\+\+\$\+\+\+', engine = 'python')

titles['movieID'].apply(lambda x: x.strip())

titles[titles['movie title'].str.contains('godfather')]
# Let's read the characters file and see whether we can find 'The Godfather' characters

characters = pd.read_csv('../input/chatbot-data/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus/movie_characters_metadata.txt'

                         , header = None, names = ['characterID', 'character name', 'movieID', 'movie title', 'gender', 'position in credits'] 

                         , sep = '\+\+\+\$\+\+\+', engine = 'python')

characters['characterID'] = characters['characterID'].str.strip()

characters['movieID'] = characters['movieID'].str.strip()

characters['gender'] = characters['gender'].str.strip().str.lower()

characters[characters['movieID'] == 'm203'].head(5)
# Gender information is available for a third of the characters

# and for those, we have twice more male than female

characters['gender'].value_counts()
# Let's read the dialog lines and see whether we can find the opening 'Godfather' scene (monologue not included unfortunately)

lines = pd.read_csv('../input/chatbot-data/cornell_movie_dialogs_corpus/cornell movie-dialogs corpus/movie_lines.txt'

                    , header = None, names = ['lineID', 'characterID',  'movieID', 'character name', 'line text'] 

                    , sep = '\+\+\+\$\+\+\+', engine = 'python')

lines['lineID'] = lines['lineID'].str.strip()

lines['characterID'] = lines['characterID'].str.strip()

lines['movieID'] = lines['movieID'].str.strip()

lines[lines['movieID'] == 'm203'].sort_values(by=['lineID']).head(5)
# Let's join character gender information to the lines table

# Note that gender information is not available for Bonasera

lines2 = lines.join(characters.set_index('characterID'), on='characterID', rsuffix='_char')

lines2.head(5)

lines2 = lines2.drop(columns = ['character name_char', 'movieID_char'])

lines2[lines2['movieID'] == 'm203'].sort_values(by=['lineID']).head(5)

# Let's remove lines with missing gender information, and replace special characters with a space

lines2 = lines2.drop(lines2[(lines2.gender != 'm') & (lines2.gender != 'f')].index)

lines2['line text'] = lines2['line text'].str.replace("[^a-zA-Z0-9 ']", " ", regex = True).fillna("")

lines2[lines2['movieID'] == 'm203'].sort_values(by=['lineID']).head(5)
# Women account for 30% of lines, men for 70%

lines2['gender'].value_counts()
# We'll use below function to transform a full sentence into a vector

#   The function first lists vectors associated to each word

#   and then returns the average vector for the full sentence

# The resulting vector has 300 dimensions (0 to 299). Each dimension has a value between -1 and +1

# In the example below, dimansion 79 has the highest value 0.16

def transform_sentence(line):

    transform_sentence_result = [model[w] for w in word_tokenize(line) if w in model]

    if len(transform_sentence_result) == 0:

        return np.array([0.0]*300)

    else:

        return pd.Series(transform_sentence_result).mean(axis=0)

pd.Series(transform_sentence('Bonasera we know each other for years')).sort_values()
# Which word contributes most to the sentence?

# In our example, dimension 79 has the highest value 0.16

# Which word contributes most to that dimension 79?

def most_significant_word(line):

    # Which dimension "a" has the highest value?

    a = pd.Series([model[w] for w in word_tokenize(line) if w in model]).mean(axis=0).argmax()

    most_significant_word = ''

    # Which word contributes most to that dimension?

    significance = 0

    for w in word_tokenize(line):

        if w in model:

            if model[w][a] > significance:

                most_significant_word = w

                significance = model[w][a]

    return most_significant_word

most_significant_word('Bonasera we know each other for years')
# The example below for the word 'Corleone' shows that

# dimension 125 has the minimum value -0.68

# dimension 61 has the maximum value 0.60

# mean value is close to zero

pd.DataFrame(model['Corleone']).agg(['min', 'idxmin', 'mean', 'max', 'idxmax'])
# Let's see how lines 104504 & 104505 look like once vectorized

lines3 = lines2.apply(lambda x: transform_sentence(x['line text']), axis=1, result_type='expand')

lines3 = lines3[lines3.max(axis=1) != 0]

lines3 = lines3.join(lines2['gender'])

lines3.loc[104504:104506,:]
# Let's fit a logistic regression classifier with the vectorized sentences

X = lines3.drop(['gender'], axis=1)

y = lines3['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60)

lr = LogisticRegression(solver='lbfgs', class_weight='balanced')

lr.fit(X_train, y_train)

# Function below provides a report including accuracy score, confusion matrix and classification report

def report(clf, X, y):

    acc = accuracy_score(y_true=y, 

                         y_pred=clf.predict(X))

    cm = pd.DataFrame(confusion_matrix(y_true=y, 

                                       y_pred=clf.predict(X)), 

                      index=clf.classes_, 

                      columns=clf.classes_)

    rep = classification_report(y_true=y, 

                                y_pred=clf.predict(X))

    return '{:.3f}\n\n{}\n\n{}'.format(acc, cm, rep)

# Let's print the report

print()

print('Logistic Regression')

print(report(lr, X_test, y_test))

for strategy in ['stratified', 'most_frequent', 'uniform']:

    dummy = DummyClassifier(strategy=strategy)

    dummy.fit(X_train, y_train)

    print('')

    print('dummy', strategy)

    print(report(dummy, X_test, y_test))
# Let's see how our two lines 104504 & 104506 look like once counted (1 if the word is present, 0 if not)

corpus = lines2['line text']

vectorizer = CountVectorizer(binary=True, min_df=0.001, max_df=0.999)

vectorizer.fit(corpus)

X2 = pd.DataFrame(vectorizer.transform(corpus).A, columns = vectorizer.get_feature_names(), index=lines2.index)

y2 = lines2['gender']

X2.loc[104504:104506,:]
# Let's fit a logistic regression classifier with the counting, and print the report

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.60)

lr2 = LogisticRegression(solver='lbfgs', class_weight='balanced')

lr2.fit(X2_train, y2_train)

print()

print('Logistic Regression - CountVectorizer')

print(report(lr2, X2_test, y2_test))
lr3 = LogisticRegression(solver='lbfgs', class_weight='balanced')

lr3.fit(X, y)

probas = pd.DataFrame(lr.predict_proba(X), 

                         columns=['P({})'.format(X) for X in lr.classes_], 

                         index=X.index)

probas = probas.join(lines2)

probas.sort_values('P(f)').head(3)
probas.sort_values('P(m)').head(3)

# No comment ;-)