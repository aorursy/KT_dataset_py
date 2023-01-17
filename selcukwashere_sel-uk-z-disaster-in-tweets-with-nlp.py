import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
count_vectorizer = feature_extraction.text.CountVectorizer() # this is our Vectorizer model.
## In this cell, we get the counts for the first 5 tweets in our training data.

exp_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
## We use .todense() to turn sparse into dense.

## The reason for this is because sparse only keeps non-zero indexes to save space.

print(exp_train_vectors[0].todense().shape)

print(exp_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])
## Since our vectors are too big, we want to push our model weights

## towards 0 without decreasing all different words.

## Ridge Regression is an effective way for this.

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors,train_df["target"]) ## we fit our training data and vectors
test_pred = pd.Series(clf.predict(test_vectors),name="target").astype(int)

results = pd.concat([test_df["id"],test_pred],axis=1)

results.head()
results.to_csv("submission.csv",index=False)