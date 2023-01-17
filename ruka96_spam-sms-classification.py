import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

%matplotlib inline
df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
# Drop some columns
df = df[['v1', 'v2']]
# Randomize the data set
np.random.shuffle(df.values)
df.head()
from sklearn.feature_extraction.text import *
# Count
count_vectorize = CountVectorizer()
c_vectors = count_vectorize.fit_transform(df['v2'])

# tf-idf
tf_idf_vect = TfidfVectorizer()
tf_idf_vectors = tf_idf_vect.fit_transform(df['v2'])
# Use scikit-learn to tokenize
analyze = count_vectorize.build_analyzer()
print (df['v2'][0])
print (analyze(df['v2'][0]))
## GloVe 100 dimensions
with open("../input/glove6b100dtxt/glove.6B.100d.txt", "r") as lines:
    glove_100d = { line.split()[0]:  np.float_(line.split()[1:]) for line in lines }
class MeanEmbeddingVectorizer(object):
    def __init__(self, glove):
        self.glove = glove
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(glove.items())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.glove[w] for w in words if w in self.glove]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
# tokenize the input
lines = np.array([ analyze(line) for line in df['v2'] ])
# Prepare word embedded
word_embedded_100d = MeanEmbeddingVectorizer(glove_100d)
# Extract
word2vec_100d = word_embedded_100d.transform(lines)
X = c_vectors
y = df['v1']
X_train = X[0:3900]
X_test = X[3901: 5571]
y_train = df['v1'][0:3900]
y_test = df['v1'][3901: 5571]
# Naive Bayes Classification
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
nb_clf = MultinomialNB(alpha=1)
nb_clf = nb_clf.fit(X_train, y_train)
prediction = nb_clf.predict(X_test)
print (classification_report(y_test, prediction))
X = tf_idf_vectors
y = df['v1']
X_train = X[0:3900]
X_test = X[3901: 5571]
y_train = df['v1'][0:3900]
y_test = df['v1'][3901: 5571]
nb_clf = MultinomialNB(alpha=1)
nb_clf = nb_clf.fit(X_train, y_train)
prediction = nb_clf.predict(X_test)
print (classification_report(y_test, prediction))
N = 5572
D = 100
X = word2vec_100d
y = df['v1']
# Preparing
X = np.zeros(shape=(N,D))
for i in range(N):
    for j in range(D):
        X[i][j] = word2vec_100d[i][j]
# Split Data
X_train = X[0:3900]
X_test = X[3901: 5571]
y_train = df['v1'][0:3900]
y_test = df['v1'][3901: 5571]
# Decision Tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print (classification_report(y_test, prediction))
