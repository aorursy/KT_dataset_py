import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
## creates a dictionary with (# of tweet, # position of word in alphabetically ordered list over whole set) # occurence of word in tweet

count_vectorizer = feature_extraction.text.CountVectorizer()

Tfidf = feature_extraction.text.TfidfVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)(sparse - mostly zeros, dense matrix - non zeroes)

print(example_train_vectors.todense().shape, "<-- in the first 5 tweets there are 54 unique words (tokens)")

print(example_train_vectors[0].todense(),"<-- representation of the 13 words in the first tweet in the list of 54 tokens")
#train_vectors = Tfidf.fit_transform(train_df["text"])

#test_vectors = Tfidf.transform(test_df["text"])



train_vectors = count_vectorizer.fit_transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])



## note that we're NOT using .fit_transform() for the test. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.
## Because the vectors are big, we risk running into a high variance scenario. We introduce ridge regression to reduce this. 

# Ridge regression penalizes big weights CostRR = Cost + (alpha * weights^2) and can therefore reduce the chance of overfitting the training data

clf = linear_model.RidgeClassifier(alpha = 5.5)

#clf = SGDClassifier(loss="log", max_iter=500, alpha = 0.001)
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

print(sum(scores/3))

clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = clf.predict(test_vectors)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)