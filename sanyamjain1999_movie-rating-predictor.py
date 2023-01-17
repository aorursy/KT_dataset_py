import numpy as np

import pandas as pd
d1 = pd.read_csv("../input/Sample_submission.csv")

Xt = pd.read_csv("../input/Test.csv")

d2 = pd.read_csv("../input/Train.csv") 
d2.head()
d1.head()
Xt.head()
df = d2.values
x = df[:, 0]

print(x[:2])

print(x.shape)
y = df[:, 1]

print(y[:2])

print(y.shape)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

y
from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')

en_stopwords = set(stopwords.words('english'))

ps = PorterStemmer()
def getCleanReview(review):

    

    review = review.lower()

    review = review.replace("<br /><br />"," ")

    

    #Tokenize

    tokens = tokenizer.tokenize(review)

    new_tokens = [token for token in tokens if token not in en_stopwords]

    stemmed_tokens = [ps.stem(token) for token in new_tokens]

    

    cleaned_review = ' '.join(stemmed_tokens)

    

    return cleaned_review
x = [getCleanReview(i) for i in x]
print(x[:2])
xt = Xt.values

xt = xt.reshape(-1, )

xt.shape
xt[:2]
xt = [getCleanReview(i) for i in xt]
xt[:2]
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(ngram_range=(2, 2))
tf.fit(x)

x = tf.transform(x)

xt = tf.transform(xt)
from sklearn.naive_bayes import MultinomialNB
mb = MultinomialNB()

mb.fit(x,y)
pred = mb.predict(xt)
d1['label'] = ['pos' if each == 1 else 'neg' for each in pred ]
d1.to_csv('submission.csv', index=None)