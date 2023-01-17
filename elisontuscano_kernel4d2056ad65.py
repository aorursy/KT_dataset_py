import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test= pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

train.head(5)
from scipy import stats

train.groupby(['target']).size()

import seaborn as sns

import matplotlib.pyplot as plt

dis = pd.concat([train, test], ignore_index=True)

graph = dis.target.value_counts().values

graph_values = [(graph[1] / sum(graph)),  (graph[0] / sum(graph))]

sns.barplot(x=['Disaster', 'Not Disaster'], y=graph_values, palette="hls").set_title('Target distribution data')

plt.show()
sns.barplot(y=train[dis.target == 1].location.value_counts()[:10].index, 

            x=train[dis.target == 1].location.value_counts()[:10].values,

            palette="hls").set_title('Top 10 Locations in Real Disaster')

plt.show()
sns.barplot(y=train[dis.target == 0].location.value_counts()[:10].index, 

            x=train[dis.target == 0].location.value_counts()[:10].values,

            palette="hls").set_title('Top 10 Locations in Non Disaster')

plt.show()
count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train["text"][0:5])

## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
count_vectorizer = feature_extraction.text.CountVectorizer()

train_vectors = count_vectorizer.fit_transform(train["text"])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=3, scoring="f1")

scores
tfidf_count_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

train_vectors = tfidf_count_vectorizer.fit_transform(train["text"])

test_vectors = tfidf_count_vectorizer.transform(test["text"])
clf = linear_model.RidgeClassifier()

scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=3, scoring="f1")

scores
svc = SVC(kernel = 'linear', random_state = 0)

scores = model_selection.cross_val_score(svc, train_vectors, train["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, train["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)