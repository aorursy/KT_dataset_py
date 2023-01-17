



import numpy as np

import pandas as pd

from sklearn import feature_extraction, linear_model, model_selection, preprocessing



train_df = pd.read_csv("../input/nlp-getting-started/train.csv")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

print(train_df.shape)

print(test_df.shape)

print(submission.shape)
count_vectorizer = feature_extraction.text.CountVectorizer()

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])

#

clf = linear_model.RidgeClassifier()

#

scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

print(scores)
clf.fit(train_vectors, train_df["target"])
submission["target"] = clf.predict(test_vectors)

submission.to_csv("submission_real_or_not.csv", index=False)

print(submission.shape)