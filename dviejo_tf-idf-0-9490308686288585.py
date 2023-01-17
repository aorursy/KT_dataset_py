# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/spam.csv", encoding='latin-1')
df.v1 = df.v1.astype("category")
train_x, test_x, train_y, test_y = train_test_split(df.v2, df.v1, test_size=0.25, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
vector = vectorizer.fit(train_x)
train_x = vectorizer.transform(train_x)
classifier = LogisticRegression(solver='liblinear')

classifier.fit(train_x, train_y)

test_x = vectorizer.transform(test_x)
predicted = classifier.predict(test_x)

acc = accuracy_score(test_y, predicted)

print(acc)

print("Accuracy: %0.2f percent" % (acc))
def predict_email(text):

    p = classifier.predict(vectorizer.transform([text]))

    print(p[0])
predict_email("-PLS STOP bootydelious (32/F) is inviting you to be her friend. Reply YES-434 or NO-434 See her: www.SMS.ac/u/bootydelious STOP? Send STOP FRND to 62468")