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
# Load data set

train = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

train.shape

train.head()

train['sentiment'].value_counts()
# Split into training and validation set



from sklearn.model_selection import train_test_split



xtrain, xvalid, ytrain, yvalid = train_test_split(train.review.values, train.sentiment.values, 

                                                  random_state=42, 

                                                  test_size=0.2, shuffle=True)



xtrain
# Remove HTML tags

from bs4 import BeautifulSoup



print(train["review"][0])

print(BeautifulSoup(train["review"][0]).get_text())



def prep_remove_html(text):

    return BeautifulSoup(text).get_text().lower()
# Remove punctuation, numbers etc

import re



def prep_only_words(text):

    return re.sub("[^a-zA-Z]", " ", text).split()
# Remove stop words

import nltk

from nltk.corpus import stopwords



def prep_remove_stopwords(word_list):

    return [w for w in word_list if not w in stopwords.words("english")]
# Complete text preprocessing



def preprocess(text):

    return " ".join(prep_remove_stopwords(prep_only_words(prep_remove_html(text))))
# Run preprocessing - nope



#train["cleaned_text"] = train["review"].apply(preprocess)

#train.head
# All-in-one vectorizer



from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(max_features=5000, 

                             ngram_range=(1, 3), 

                             stop_words = stopwords.words('english'), 

                             preprocessor = prep_remove_html, 

                             strip_accents='unicode')



vectorizer.fit(list(xtrain) + list(xvalid))
xtrain_features = vectorizer.transform(xtrain)

xvalid_features = vectorizer.transform(xvalid)

#for name, count in zip(vectorizer.get_feature_names(), np.sum(xtrain_features.toarray(), axis=0)):

#    print(name, count)



# Score simple model



from sklearn.linear_model import LogisticRegression



clf_lr = LogisticRegression()

clf_lr.fit(xtrain_features, ytrain)

clf_lr.score(xvalid_features, yvalid)

# ... and another one



from sklearn.naive_bayes import BernoulliNB



clf_nb = BernoulliNB()

clf_nb.fit(xtrain_features, ytrain)

clf_nb.score(xvalid_features, yvalid)

# Random Forest



from sklearn.ensemble import RandomForestClassifier



clf_rf = RandomForestClassifier(n_estimators = 100)

clf_rf.fit(xtrain_features, ytrain)

clf_rf.score(xvalid_features, yvalid)
# Predict test data



test = pd.read_csv("/kaggle/input/word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", quoting=3)

test_features = vectorizer.transform(test.review.values)

predictions = clf_lr.predict(test_features)

output = pd.DataFrame( data={"id":test["id"], "sentiment":predictions} )

output.to_csv( "tfidf-logreg.csv", index=False, quoting=3 )