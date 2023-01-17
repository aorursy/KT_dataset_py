import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.sparse import vstack

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

from nltk.corpus import LazyCorpusLoader, CategorizedPlaintextCorpusReader
# Listing the first 5 documents's ID and categories
!head -n 5 '/kaggle/input/reuters/reuters/reuters/cats.txt'
# https://www.kaggle.com/alvations/testing-1000-files-datasets-from-nltk
reuters = LazyCorpusLoader('reuters', CategorizedPlaintextCorpusReader, 
                           '(training|test).*', cat_file='cats.txt', encoding='ISO-8859-2',
                          nltk_data_subdir='/kaggle/input/reuters/reuters/reuters/')
# https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/
reuters.words()
# See how many categories are there? 
num_of_cat = len(reuters.categories())
print('Number of categories: ' + str(num_of_cat) + ' categories')
reuters.categories()   # List all available categories.
reuters.fileids("jobs")[:5]
reuters.words(reuters.fileids("jobs")[0])
# List the raw text of the first 5 articles. 
for i in range(5):
    print('Article #' + str(i+1))
    print(reuters.raw(reuters.fileids("jobs")[i]))
    
reuters.categories(reuters.fileids("jobs")[:20])
# File IDs
reuters.fileids()[:5]
train_docs = list(filter(lambda doc: doc.startswith("train"),
                        reuters.fileids()));
print('Number of docs in the test set: ' + str(len(train_docs)))
train_docs[:5]
test_docs = list(filter(lambda doc: doc.startswith("test"),
                        reuters.fileids()));
print('Number of docs in the test set: ' + str(len(test_docs)))
test_docs[:5]
train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])

# All documents in The Reuters dataset
whole_docs, whole_cats = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if True])
whole_cats[:5]
train_documents[0]
train_categories[:20]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words = 'english')

vectorised_train_documents = vectorizer.fit_transform(train_documents)
vectorised_test_documents = vectorizer.transform(test_documents)

vectorizer2 = TfidfVectorizer(stop_words = 'english')
vec_whole_docs = vectorizer2.fit_transform(whole_docs)
vectorised_train_documents
vec_whole_docs
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_categories)
test_labels = mlb.transform(test_categories)

# For the whole dataset
mlb2 = MultiLabelBinarizer()
whole_labels = mlb2.fit_transform(whole_cats)
train_labels.shape
whole_labels.shape   # Training docs = 7769 docs. 
# Create the whole dataset from training-only vectorized matrix. 
print('X stuff: ')
print(vectorised_train_documents.shape)
print(vectorised_test_documents.shape)
print(type(vectorised_train_documents))
X = vstack([vectorised_train_documents, vectorised_test_documents])
print(X.shape)

print('Label stuff: ')
print(train_labels.shape)
print(test_labels.shape)
y = np.concatenate((train_labels, test_labels))
y.shape
# https://towardsdatascience.com/multi-class-text-classification-with-sklearn-and-nltk-in-python-a-software-engineering-use-case-779d4a28ba5
from sklearn.ensemble import RandomForestClassifier
clf_random = RandomForestClassifier()
# random_clf.fit(X, y)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

clf_svm = OneVsRestClassifier(LinearSVC())
# clf_svm.fit(vec_whole_docs, whole_labels)

from sklearn.neural_network import MLPClassifier
clf_mlp1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1,), random_state=1)
clf_mlp2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2,), random_state=1)
clf_mlp3 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3,), random_state=1)
# clf_mlp.fit(vec_whole_docs, whole_labels)
# Setup 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
# scoring = ['precision_samples', 'recall_samples', 'f1_samples']
# scoring = ['precision_macro', 'recall_macro', 'f1_macro']       # No class imbalance consideration
scoring = ['precision_micro', 'recall_micro', 'f1_micro']
cv = 3
scores = cross_validate(clf_random, X, y, cv=cv, scoring=scoring)

# Whole dataset (begin before vectorization)
scores_whole = cross_validate(clf_random, vec_whole_docs, whole_labels, cv=cv, scoring=scoring)
# print the report 
scores
scores_whole
# SVM 
scores_svm = cross_validate(clf_svm, vec_whole_docs, whole_labels, cv=cv, scoring=scoring)
scores_svm
# MLP
scores_mlp1 = cross_validate(clf_mlp1, vec_whole_docs, whole_labels, cv=cv, scoring=scoring)
scores_mlp2 = cross_validate(clf_mlp2, vec_whole_docs, whole_labels, cv=cv, scoring=scoring)
scores_mlp3 = cross_validate(clf_mlp3, vec_whole_docs, whole_labels, cv=cv, scoring=scoring)
scores_mlp1
scores_mlp2
scores_mlp3
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(clf_mlp3, vec_whole_docs, whole_labels, train_sizes=[2000, 3000, 7000], cv=5)
train_scores
valid_scores
import pandas as pd
import numpy as np
import gzip

# Read the JSON file. 
def parse(path):
    g = open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    data = {}
    for d in parse(path):
        data[i] = d
        i += 1
    return pd.DataFrame.from_dict(data, orient='index')

DATASET_NAME = '../input/appsforandroid/reviews_Apps_for_Android_5.json'
df = getDF(DATASET_NAME)

df.head(30)
df.dropna(inplace=True)
df[df['overall'] != 3]
df['Positivity'] = np.where(df['overall'] > 3, 1, 0)
df['intRating'] = df['overall'].astype(int)
df.head(20)
# X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Positivity'], random_state = 0)

# from sklearn.preprocessing import MultiLabelBinarizer
# mlb = MultiLabelBinarizer()

apps_X = df['reviewText']
apps_y = pd.get_dummies(df['Positivity'])
apps_X
apps_y[:3]
# TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words = 'english')
vect_bigram = TfidfVectorizer(stop_words = 'english', ngram_range = (1,2))
apps_X_vec = vect.fit_transform(apps_X)
apps_X_vec_bigram = vect_bigram.fit_transform(apps_X)
print(len(vect.get_feature_names()))
print(len(vect_bigram.get_feature_names()))
apps_X_vec.shape
apps_y.shape
apps_y_multi = pd.get_dummies(df['overall'])
apps_y_multi
# Models
from sklearn.linear_model import LogisticRegression
m_lr = LogisticRegression()
# m_lr.fit(apps_X_vec, df['Positivity'])

from sklearn.ensemble import RandomForestClassifier
m_random = RandomForestClassifier()
# m_random.fit(apps_X_vec, df['Positivity'])

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
m_svm = OneVsRestClassifier(LinearSVC())
# Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
# scoring = ['precision_samples', 'recall_samples', 'f1_samples']
# scoring = ['precision_macro', 'recall_macro', 'f1_macro']
scoring = ['precision_micro', 'recall_micro', 'f1_micro']
cv = 3

apps_scores = cross_validate(m_lr, apps_X_vec, df['Positivity'], cv=cv, scoring=scoring)
apps_scores_bigram = cross_validate(m_lr, apps_X_vec_bigram, df['Positivity'], cv=cv, scoring=scoring)
apps_scores
apps_scores_bigram
apps_scores_multi = cross_validate(m_lr, apps_X_vec, df['overall'], cv=cv, scoring=scoring)
apps_scores_bigram_multi = cross_validate(m_lr, apps_X_vec_bigram, df['overall'], cv=cv, scoring=scoring)
apps_scores_multi
apps_scores_bigram_multi
# Random forest CV
# apps_scores_random = cross_validate(m_random, apps_X_vec, df['Positivity'], cv=cv, scoring=scoring)   # Too long training time
# apps_scores_random
# SVM CV (RandomForest takes too much time to train with this dataset)
apps_scores_svm = cross_validate(m_svm, apps_X_vec, df['Positivity'], cv=cv, scoring=scoring)
apps_scores_svm
apps_scores_svm_bigram = cross_validate(m_svm, apps_X_vec_bigram, df['Positivity'], cv=cv, scoring=scoring)   # Too long training time
apps_scores_svm_bigram
# SVM Multiclass Unigram 
apps_scores_svm_multi = cross_validate(m_svm, apps_X_vec, df['intRating'], cv=cv, scoring=scoring)
apps_scores_svm_multi
# SVM Multiclass Bigram 
apps_scores_svm_bigram = cross_validate(m_svm, apps_X_vec_bigram, df['intRating'], cv=cv, scoring=scoring)  
apps_scores_svm_bigram
# What if we calculate using "macro"? 
scoring = ['precision_macro', 'recall_macro', 'f1_macro']
# scoring = ['precision_micro', 'recall_micro', 'f1_micro']
apps_scores_svm_bigram = cross_validate(m_svm, apps_X_vec_bigram, df['Positivity'], cv=cv, scoring=scoring)
apps_scores_svm_bigram