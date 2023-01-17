import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
#Create instance

count_vectorizer = feature_extraction.text.CountVectorizer()
#Fit Transform train data

train_vectors = count_vectorizer.fit_transform(train_df["text"])
#Only transform test data - so that train and test use the same vectors

test_vectors = count_vectorizer.transform(test_df["text"])
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores.mean()
clf.fit(train_vectors, train_df["target"])
#get sample file for creating submission file

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
#Got to the Output section of this Kernel -> click on Submit to Competition

sample_submission.to_csv("submission.csv", index=False)