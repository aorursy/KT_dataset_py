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
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
df = pd.read_csv("../input/taylorvsshakes.csv")
df.head()
# Split the data into training and testing data

X_train, X_test, y_train, y_test = train_test_split(

                                             df['Text'], df['Author'], 

                                             test_size=0.33)
# Create CountVectorizer object to turn text into bag-of-words vectors 

# Remove stopwords

count_vectorizer = CountVectorizer(stop_words='english')



# count_vectorizer turns training and testing data into bag-of-words vectors



# fit_transform means that count_vectorizer learns the dictionary from training data

count_train = count_vectorizer.fit_transform(X_train.values)



# transform means that count_vectorizer uses the same dictionary as the training data

# unfamiliar words thrown out

count_test = count_vectorizer.transform(X_test.values)



# Prints count_vectorizer's dictionary with word counts

# print(count_vectorizer.vocabulary_)



# Create the CountVectorizer DataFrame: count_df

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

count_df.head()
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



nb_classifier = MultinomialNB()

nb_classifier.fit(count_train, y_train)



pred = nb_classifier.predict(count_test)

metrics.accuracy_score(y_test, pred)
print("Columns: Predicted Taylor, Predicted Shakespeare")

print("Rows: Actual Taylor, Actual Shakespeare")

metrics.confusion_matrix(y_test, pred, labels=["TAYLOR", "SHAKES"])