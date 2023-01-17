import pandas as pd

import numpy as np

from sklearn.cross_validation import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.cross_validation import KFold, cross_val_score

from sklearn.naive_bayes import MultinomialNB



path = '../input/Combined_News_DJIA.csv'

data = pd.read_csv(path)

data.head()
# Create feature matrix (X) and the response vector (y)

X = data.iloc[:, 2:27]

y = data.Label
# train_test_split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.19, random_state=13)
# the length of y_test is the same as testing data suggested by the dataset's creator

len(y_test)
# combine all the headlines into a string per each row and add them to trainheadlines

trainheadlines = []

for row in range(0, len(X_train.index)):

    trainheadlines.append(' '.join(str(x) for x in X_train.iloc[row, 0:25]))
# instantiate and fit the CountVectorizer

vect = CountVectorizer()

vect_train = vect.fit_transform(trainheadlines)

print(vect_train.shape)
# instantiate and fit the LogisticRegression model

logreg = LogisticRegression()

logreg.fit(vect_train, y_train)
# follow the same steps for the testing data as the training data

testheadlines = []

for row in range(0, len(X_test.index)):

    testheadlines.append(' '.join(str(x) for x in X_test.iloc[row, 0:25]))

vect_test = vect.transform(testheadlines)

predictions = logreg.predict(vect_test)
# calculate accuracy

metrics.accuracy_score(y_test, predictions)
# Use crosstab to look at the results

pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
# create vector transforms of X

headlines = []

for x in range(0, len(X.index)):

    headlines.append(' '.join(str(x) for x in X.iloc[row, 0:25]))

vect_headlines = vect.transform(headlines)
# split the dataset using K-folds with shuffle=False

# calculate the cross_val_scores

kf = KFold(len(y), n_folds=10, shuffle=False)

print(cross_val_score(logreg, vect_headlines, y, cv=kf))

print(cross_val_score(logreg, vect_headlines, y, cv=kf).mean())
# split the dataset using K-folds with shuffle=True

kf = KFold(len(y), n_folds=10, shuffle=True)

print(cross_val_score(logreg, vect_headlines, y, cv=kf))

print(cross_val_score(logreg, vect_headlines, y, cv=kf).mean())
# Building and evaluating a model

# import and instantiate and fit a Multinominal Naive Bayes model

nb = MultinomialNB(alpha=1.0)

%time nb.fit(vect_train, y_train)
# make class predictions for vect_test

predictions = nb.predict(vect_test)
# calculate accuracy of class predictions

metrics.accuracy_score(y_test, predictions)
# print the confusion matrix

metrics.confusion_matrix(y_test, predictions)