# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# First we need to load the data into a dataframe



df = pd.read_csv("../input/train.csv")
df.head()
# Lets also have a look at the test data



tdf = pd.read_csv("../input/test.csv")

print(tdf.head())

del(tdf)
# Let's first have a look at some of the comments whose scores were above 0.0



random_indices = np.random.choice([i for i in range(len(df)) if df["target"][i] > 0.], 5)

for i in random_indices:

    print("Text: ", df["comment_text"][i])

    print("Score: ", df["target"][i])
# Now let's have a look at completely non toxic comments



random_indices = np.random.choice([i for i in range(len(df)) if df["target"][i] == 0.], 5)

for i in random_indices:

    print("Text: ", df["comment_text"][i])

    print("Score: ", df["target"][i])
# Now lets have a look at very toxic comments: target > 0.75



random_indices = np.random.choice([i for i in range(len(df)) if df["target"][i] >= .75], 5)

for i in random_indices:

    print("Text: ", df["comment_text"][i])

    print("Score: ", df["target"][i])
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 5))

plt.hist(df['target'], bins = 100)

plt.show()
plt.figure(figsize=(20, 5))

plt.hist(df[df['target'] > 0.0]['target'], bins = 100)

plt.show()
tdf = df.loc[:30000, ["comment_text", "target"]]

tdf.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(tdf["comment_text"], tdf["target"], test_size = .10)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
cvect = CountVectorizer(min_df = 0.1, ngram_range=(1, 3), analyzer="word").fit(X_train)

X_trcv = cvect.transform(X_train)

X_tscv = cvect.transform(X_test)



# In order to convert the coninuous values of the target to binary, as logistic regression can accept only binary values (0 or 1) as the target values

# Here we choose 0.1 as a cutoff as we want to classify even slightly toxic comments as toxic

y_train_lg = np.array(y_train > 0.1, dtype=np.float)

y_test_lg = np.array(y_test > 0.1, dtype=np.float)



from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_trcv, y_train_lg)

print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))

print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))

predicted = clf.predict(X_tscv)



from sklearn.metrics import precision_score, recall_score

print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))

print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))
from sklearn.dummy import DummyClassifier



dclf = DummyClassifier(strategy="most_frequent").fit(X_trcv, y_train_lg)

print("Training Accuracy: {}".format(dclf.score(X_trcv, y_train_lg)))

print("Testing Accuracy: {}".format(dclf.score(X_tscv, y_test_lg)))

predicted = dclf.predict(X_tscv)



print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))

print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))
from sklearn.naive_bayes import BernoulliNB



clf = BernoulliNB().fit(X_trcv, y_train_lg)

print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))

print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))

predicted = clf.predict(X_tscv)



from sklearn.metrics import precision_score, recall_score

print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))

print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))
from sklearn.naive_bayes import MultinomialNB



clf = MultinomialNB().fit(X_trcv, y_train_lg)

print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))

print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))

predicted = clf.predict(X_tscv)



from sklearn.metrics import precision_score, recall_score

print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))

print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))
tfvect = TfidfVectorizer().fit(X_train)



X_trtf = tfvect.transform(X_train)

X_tstf = tfvect.transform(X_test)



clf = MultinomialNB().fit(X_trtf, y_train_lg)

print("Training Accuracy: {}".format(clf.score(X_trtf, y_train_lg)))

print("Testing Accuracy: {}".format(clf.score(X_tstf, y_test_lg)))

predicted = clf.predict(X_tstf)



from sklearn.metrics import precision_score, recall_score

print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))

print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))
from sklearn.svm import SVC



clf = SVC(kernel="linear").fit(X_trtf, y_train_lg)

print("Training Accuracy: {}".format(clf.score(X_trtf, y_train_lg)))

print("Testing Accuracy: {}".format(clf.score(X_tstf, y_test_lg)))

predicted = clf.predict(X_tstf)



from sklearn.metrics import precision_score, recall_score

print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))

print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))
clf = SVC(kernel="linear").fit(X_trcv, y_train_lg)

print("Training Accuracy: {}".format(clf.score(X_trcv, y_train_lg)))

print("Testing Accuracy: {}".format(clf.score(X_tscv, y_test_lg)))

predicted = clf.predict(X_tscv)



from sklearn.metrics import precision_score, recall_score

print("Test Precision: {}".format(precision_score(y_test_lg, predicted)))

print("Test Recall: {}".format(recall_score(y_test_lg, predicted)))