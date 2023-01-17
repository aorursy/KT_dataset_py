# Import the required packages

import pandas as pd
import numpy as np
# Read the training dataset

train = pd.read_csv('../input/nlp-getting-started/train.csv')
train.head()
# Read the test dataset

test = pd.read_csv('../input/nlp-getting-started/test.csv')
test.head()
# Check the shape , info and distribution of the target variable in the train dataset

print(train.shape)
# Data constitutes of 5 coolumns and 7613 entries


print('---------------------\n')
print(train.info())
# Text column has no null value

print('---------------------\n')
print(train.target.value_counts())
# Target variable distribution
# 0    4342
# 1    3271
# Creating a subset of the dataset containing only the text column and the target

train_text = train[['text','target']]
train_text.head()
# Initiating a vectorizer

# Importing CountVectorizer from sklearn feature extraction package
from sklearn.feature_extraction.text import CountVectorizer

# Storing the vectorizer in a variable
vectorizer = CountVectorizer()

# Transforming the text column based on the values fit on the train data
train_features = vectorizer.fit_transform(train_text.text)

# Transforming the test features on the vectorizer trained on the training documents
test_features = vectorizer.transform(test.text)

# You can get the vocabulary from the training set
# vocab = vectorizer.vocabulary_
# print(vocab.)
# print("-----------------\n")

# Checking a sample train set
print(train_features[0].todense())
# X and y variables for training a model

X = train_features

y = train_text.target
# Model building

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 4000)
# Scores
from sklearn import model_selection

scores = model_selection.cross_val_score(model, X, y,cv = 3, scoring = "f1")
np.mean(scores)
# Predictions on the training set

model.fit(train_features,train_text.target)
# Read the sample submissions
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission.head()
# My submissions
sample_submission['target'] = model.predict(test_features)
sample_submission.head()
# Exporting the csv file for the sample submissions

sample_submission.to_csv("My_submissions.csv")