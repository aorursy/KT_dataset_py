import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]
count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)