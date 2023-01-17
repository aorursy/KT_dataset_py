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
train= pd.read_csv('../input/labeledTrainData.tsv',header=0,delimiter='\t',quoting =3)
train.head()
train.shape
train.columns.values



print(train.review[0])
from bs4 import BeautifulSoup

example1 = BeautifulSoup(train.review[0])

print(example1.get_text())
import re

letter_only = re.sub("[^a-zA-Z]"," ",example1.get_text())
print(letter_only)
lower_case = letter_only.lower()

words = lower_case.split()

print(type(words))
words[0]
import nltk

from nltk.corpus import stopwords
print(stopwords.words("english"))
words = [w for w in words if not w in stopwords.words("english")]

print(words)
stop = set(stopwords.words("english"))



def review_words(raw_review):

    review_txt = BeautifulSoup(raw_review)

    letter_only = re.sub("[^a-zA-Z]"," ",review_txt.get_text())

    words = letter_only.lower().split()

    meaningfl_words = [w for w in words if not w in stop]

    return(" ".join(meaningfl_words))

num_reviews = train.review.size
clean_reviews = []

for i in range(0,num_reviews):

    clean_reviews.append(review_words(train.review[i]))

    if((i+1)%1000 == 0):

        print(i+1)
clean_reviews[0]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer='word',tokenizer=None,    \

                            preprocessor = None, stop_words = None, max_features = 5000)
train_data_features= vectorizer.fit_transform(clean_reviews)
train_data_features = train_data_features.toarray()
train_data_features.shape
vocab = vectorizer.get_feature_names()

print(vocab)
dist = np.sum(train_data_features,axis=0)

for tag,count in zip(vocab,dist):

    print(count,tag)
from xgboost import XGBClassifier



xgb = XGBClassifier(n_estimators = 500, learning_rate= 0.1, max_depth = 5)

xgb.fit(train_data_features, train.sentiment)
test_data = pd.read_csv('../input/testData.tsv',header=0,delimiter='\t',quoting=3)
print(test_data.head())
num_reviews = len(test_data.review)
clean_reviews = []

for i in range(0,num_reviews):

    clean_reviews.append(review_words(test_data.review[i]))
test_features = vectorizer.transform(clean_reviews)
test_features = test_features.toarray() 
result = xgb.predict(test_features)

output = pd.DataFrame(data = {'id':test_data.id,"sentiment":result})

output.to_csv("output.csv",index=False, quoting=3)