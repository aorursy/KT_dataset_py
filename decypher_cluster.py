

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import re
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# Any results you write to the current directory are saved as output.
from sklearn.ensemble import  RandomForestClassifier as RFC
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score,accuracy_score ,roc_auc_score
from nltk.stem.snowball import SnowballStemmer
import xgboost as xgb
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import LogisticRegression as LR
df = pd.read_csv("/kaggle/input/tweets/tweets.csv")
df.head()
targets = df["Sarcasm"]
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
 
def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
 
    classifier.fit(X_train, y_train)
    print ("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
 
trial1 = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB()),
])
 
train(trial1, df.tweet, df.Sarcasm)
from nltk.corpus import stopwords
 
trial2 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('classifier', MultinomialNB()),
])
 
train(trial2, df.tweet, df.Sarcasm)

trial3 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('classifier', MultinomialNB(alpha=0.05)),
])
 
train(trial3, df.tweet, df.Sarcasm)
# Accuracy: 0.909592529711
 

10
trial4 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'),
                             min_df=5)),
    ('classifier', MultinomialNB(alpha=0.05)),
])
 
train(trial4, df.tweet, df.Sarcasm)
# Accuracy: 0.903013582343
 
 
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
 
def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]
 
trial5 = Pipeline([
    ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer,
                             stop_words=stopwords.words('english') + list(string.punctuation))),
    ('classifier', MultinomialNB(alpha=0.05)),
])
 
train(trial5, df.tweet, df.Sarcasm)
import  xgboost as xgb
trial6 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'),
                             min_df=5)),
    ('classifier', xgb.XGBClassifier()),
])
 

train(trial6, df.tweet, df.Sarcasm)
# Accuracy: 0.903013582343
 
 
gb =df
gb = gb["tweet"]
modified =[]
for i in gb.values :
    url = re.sub('\#sarcasm$', '', i)
    modified.append(url)
new_df = pd.DataFrame()
new_df["target"] = df.Sarcasm
new_df["tweet"] = modified
import  xgboost as xgb
trial7 = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english'),
                             min_df=5)),
    ('classifier', xgb.XGBClassifier()),
])
 

train(trial7, new_df.tweet, new_df.target)
# Accuracy: 0.903013582343
 
 
for i in new_df["tweet"].values:
    print(i)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
X = df["Sarcasm"]
X = np.array(X)
X = df.tweet.values
X
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
print(vectorizer.get_feature_names())
import numpy as np
from numpy.linalg import norm


class Kmeans:
    '''Implementing Kmeans algorithm.'''

    def __init__(self, n_clusters, max_iter=100, random_state=123):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initializ_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initializ_centroids(X)
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
    
    def predict(self, X):
        distance = self.compute_distance(X, old_centroids)
        return self.find_closest_cluster(distance)

km = KMeans(n_clusters=2, max_iter=5)
km.fit(X.reshape(-1,1))
# centroids = km.centroids


# Plot the clustered data
# fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(X[km.labels == 0], X[km.labels == 0],
            c='green', label='cluster 1')
plt.scatter(X[km.labels == 1], X[km.labels == 1],
            c='blue', label='cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 0], marker='*', s=300,
            c='r', label='centroid')
plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of clustered data', fontweight='bold')
ax.set_aspect('equal')

