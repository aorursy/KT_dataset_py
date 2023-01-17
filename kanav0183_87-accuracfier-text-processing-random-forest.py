# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

spam_df = pd.read_csv('../input/SPAM text message 20170820 - Data.csv')

# Any results you write to the current directory are saved as output.

spam_df.head()
from wordcloud import WordCloud
spam_list = spam_df[spam_df["Category"] == "spam"]["Message"].unique().tolist()

spam_list[:2]
spam = " ".join(spam_list)

spam[:100]

spam_wordcloud = WordCloud().generate(spam)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

plt.imshow(spam_wordcloud)

plt.show()
# import the vectorizer

from sklearn.feature_extraction.text import CountVectorizer



# create an instance

count_vect = CountVectorizer()
# fit the vectorizer with data

count_vect.fit(spam_df)

# convert text to vectors

X = count_vect.transform(spam_df.Message)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



y = le.fit_transform(spam_df.Category)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='gini')
# fit the model

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

predictions.shape
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

accuracy
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=5)
clf.fit(X_train, y_train)