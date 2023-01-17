import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

path = "/kaggle/input"

for dirname,_, filenames in os.walk(path):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tweets_train_data= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

tweets_test_data= pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
print(tweets_train_data.head())

print("\n")

print(len(tweets_train_data))
print(tweets_train_data.head())

print("\n")

print(len(tweets_test_data))
tweets_train_data.isnull().sum()
tweets_test_data.isnull().sum()
modified_train_data = tweets_train_data.dropna(how='any', subset=['keyword'])

 

print("Contents of the Modified Dataframe : ")

print(modified_train_data.head())

len(modified_train_data)
modified_test_data = tweets_test_data.dropna(how='any', subset=['keyword'])

 

print("Contents of the Modified Dataframe : ")

print(modified_test_data.head())

len(modified_test_data)
print(modified_train_data.isnull().sum())

print(modified_test_data.isnull().sum())
print(modified_train_data.columns.values)

print(modified_test_data.columns.values)

print("_"*50)

modified_train_data.info()

modified_test_data.info()
count_vectorizer = feature_extraction.text.CountVectorizer()

example_train_vectors = count_vectorizer.fit_transform(modified_train_data["text"][0:5])
print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(modified_train_data["text"])

test_vectors = count_vectorizer.transform(modified_test_data["text"])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, modified_train_data["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, modified_train_data["target"])
train_vectors.shape
modified_train_data.shape
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission = sample_submission[:3237]
test_vectors
sample_submission.isnull().sum()
sample_submission.shape
test_vectors.shape
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)