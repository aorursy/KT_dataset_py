# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
cv = CountVectorizer()

nb = MultinomialNB()
spam_data = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
spam_data.head()
ham = spam_data[spam_data["Category"]=="ham"]

ham.count()
spam_data.count()
spam = spam_data[spam_data["Category"]=="spam"]

spam.count()
sns.countplot(spam_data["Category"])
spamHam_count = cv.fit_transform(spam_data["Message"])

spamHam_count.toarray()
print(cv.get_feature_names())
label = spam_data["Category"].values

label
nb.fit(spamHam_count, label)
test_sample = ["hi i will call you later","you have won a reward worth rs 1lakh!!!"]

test_sample = cv.transform(test_sample)

test_predict = nb.predict(test_sample)

test_predict
X = spamHam_count

y = label
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
nb.fit(X_train,y_train)
y_train_predict = nb.predict(X_train)

y_train_predict
score = accuracy_score(y_train,y_train_predict)

score
cm = confusion_matrix(y_train,y_train_predict)

sns.heatmap(cm, annot = True)
y_test_predict = nb.predict(X_test)

score = accuracy_score(y_test,y_test_predict)

score
cm = confusion_matrix(y_test,y_test_predict)

sns.heatmap(cm , annot = True)
print(classification_report(y_test,y_test_predict))