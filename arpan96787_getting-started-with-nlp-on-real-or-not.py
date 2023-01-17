import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]
count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 10 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])



print("Shape of train_vectors is", train_vectors.shape)

print("Shape of test_vectors is", test_vectors.shape)
from sklearn.feature_extraction.text import TfidfTransformer

tf_transformer = TfidfTransformer(use_idf=False).fit(train_vectors)

X_train_tf = tf_transformer.transform(train_vectors)

print("Shape of train_tf",X_train_tf.shape)



tf_test_vector_transformer = TfidfTransformer(use_idf=False).fit(test_vectors)

X_test_tf = tf_transformer.transform(test_vectors)

print("Shape of test_tf",X_test_tf.shape)
## Trying the Bernoulli Naive Bayes Classifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train_tf, train_df.target, test_size = 0.3)

bnb = BernoulliNB(binarize=0.0)

bnb.fit(X_train, y_train)

bnb.score(X_test, y_test)
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

#clf = linear_model.RidgeClassifier()



#Using the multinomial Naive Bayes Classifier

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tf, train_df.target)
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
scores = model_selection.cross_val_score(bnb, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
bnb.fit(X_train_tf, train_df["target"])
Bernoulli_Naive_Bayes_Submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
Bernoulli_Naive_Bayes_Submission["target"] = bnb.predict(test_vectors)
Bernoulli_Naive_Bayes_Submission.to_csv("submission_2.csv", index=False)

print("Submission file created.....")
clf.fit(X_train_tf, train_df["target"])
Naive_Bayes_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
Naive_Bayes_submission["target"] = clf.predict(test_vectors)
Naive_Bayes_submission.head()
Naive_Bayes_submission.to_csv("submission.csv", index=False)

print("Submission file created.....")