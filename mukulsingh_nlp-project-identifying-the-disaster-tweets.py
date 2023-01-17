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
train = pd.read_csv("../input/nlp-getting-started/train.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")
train.head()
print(train.shape, test.shape)
# Lets analyse what tweets are not disaster tweets

train.loc[train.target==0,"text"]
# Lets Analyze Disaster Tweets

train.loc[train.target==1,"text"].values[1:]
# Importing Count Vectorizer...

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
# Example of Count Vectorizer

text = ["Jack and Jill went up the Hill!"]

count_vectorizer.fit(text) # fit function helps learn a vocabulary about the text

transformed = count_vectorizer.transform(text) # encodes the text/doc into a vector
print(transformed.shape)

print(type(transformed))

print(transformed.toarray())
print(count_vectorizer.vocabulary_)



# Observations: Punctuations are ignored and all words are converted into Lower Case
# Test with another word

print(count_vectorizer.transform(["Jack"]).toarray()) # able to recognize the word in upper case | Location is 2

print(count_vectorizer.transform(["and"]).toarray()) # Loc is 0 as per above vocabulary

print(count_vectorizer.transform(["Jill"]).toarray())

print(count_vectorizer.transform(["Mukul Singh"]).toarray()) # No words found and hence all 0
# lets get the count of first 5 tweets

exmple  = count_vectorizer.fit_transform(train["text"][0:5])



print(exmple[0].todense().shape)

print(exmple[0].todense())
print(list(count_vectorizer.vocabulary_))

print("Unique Words are: ")

print(np.unique(list(count_vectorizer.vocabulary_)))
# Train Set

alltweets = count_vectorizer.fit_transform(train["text"]) # Transformed the Train Tweets
# Test Set

testtweets = count_vectorizer.transform(test["text"])
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import cross_val_score

ridge = RidgeClassifier()
print(cross_val_score(ridge, alltweets, train.target, cv = 5, scoring ="f1").mean())
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

gbm = GradientBoostingClassifier()

rf = RandomForestClassifier()

vc = VotingClassifier(estimators = [("rf", rf), ("ridge", ridge), ("GBM", gbm)])
vc.fit(alltweets, train.target)
solution = pd.DataFrame({"id": test.id, "target": vc.predict(testtweets)})

solution.to_csv("VC Model.csv", index=False) # Kaggle: 0.78016