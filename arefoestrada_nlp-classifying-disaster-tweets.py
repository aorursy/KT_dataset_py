import pandas as pd

import numpy as np
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train.head()
test.head()
train['target'].value_counts()
# Importing train_test_split

from sklearn.model_selection import train_test_split
# Splitting the train dataset into two disparate datasets (one to train the train dataset, the other for test the dataset)

X_train, X_test, y_train, y_test = train_test_split(train['text'], train['target'])
# Initializing the vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
# Learning the vocabulary inside the datasets and transform the train's train dataset into matrix

X_train_vect = vect.fit_transform(X_train)

X_test_vect = vect.transform(X_test)
# Importing Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB



model = MultinomialNB()
# Training the train dataset

model.fit(X_train_vect, y_train)
# Predicting the target (0 for non-disaster tweet, 1 for disaster tweet)

y_predict = model.predict(X_test_vect)
# Estimating the accuracy of the model

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# Classification report

print(classification_report(y_test, y_predict))
# Confusion matrix

print(confusion_matrix(y_test, y_predict))
# Accuracy score

print(accuracy_score(y_test, y_predict))
test.head()
# Extracting the tweets from the test dataset

text_test = test['text']
# Transforming the tweets into matrix

text_test_trans = vect.transform(text_test)
# Predicting the tweets

result = model.predict(text_test_trans)
# Putting the result into the submission's dataframe

sample_submission['target'] = result
sample_submission.head()
sample_submission.to_csv('submission.csv', index = False)