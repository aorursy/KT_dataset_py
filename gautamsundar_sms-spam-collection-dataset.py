# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
sms = pd.read_csv("../input/spam.csv", encoding = "ISO-8859-1", usecols=[0,1], skiprows=1,
                  names=["label", "message"])
sms.head()
sms.label = sms.label.map({"ham":0, "spam":1})
# more negative (ham) cases than positive (spam)
sms.label.value_counts()
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(sms.message, 
                                                                            sms.label, 
                                                                            test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer

couvec = CountVectorizer()
couvec.fit(features_train)
# number of features or tokens
trained_features = couvec.get_feature_names()
print("Number of features vectorized:", len(trained_features))
print("Examples of trained features:", trained_features[1:10])
# tokenized train documents
dtm_train = couvec.fit_transform(features_train)
print("Shape of dtm_train:", dtm_train.shape)
print(dtm_train[0:2]) # first two rows of sparse matrix
# tokenized test documents
dtm_test = couvec.transform(features_test)
print("Shape of dtm_test:", dtm_test.shape)
# import and instantiate Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# train the model
nb.fit(dtm_train, labels_train)
# make predictions
labels_pred = nb.predict(dtm_test)
# accuracy not an important metric when positive and negative classes are skewed
from sklearn import metrics
metrics.accuracy_score(labels_test, labels_pred)
# create confusion matrix to see precision and recall
metrics.confusion_matrix(labels_test, labels_pred)
# percentage of total spam detected i.e.recall
print("Recall:", metrics.recall_score(labels_test, labels_pred))
# percentage of positive (spam) predictions that are correct i.e. precision
print("Precision:", metrics.precision_score(labels_test, labels_pred))
print("Order of classes in predict_proba:", nb.classes_)
print("Example class probabilities:", nb.predict_proba(dtm_test)[0])
# since only ~15% of labels are positive (spam), a precision-recall curve is more useful than
# ROC curve
labels_prob = nb.predict_proba(dtm_test)[:, 1]
precisions, recalls, thresholds = metrics.precision_recall_curve(labels_test, 
                                                                 labels_prob)
# plotting precision-recall curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))
plt.plot(precisions[:-1], recalls[:-1])
plt.xlabel("Recalls")
plt.xticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Precisions")
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title("Precision-Recall curve")
plt.show()