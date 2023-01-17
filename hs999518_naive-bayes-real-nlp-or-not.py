import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from nltk import word_tokenize

import nltk

import string



tr=pd.read_csv("../input/nlp-getting-started/train.csv")

ts=pd.read_csv("../input/nlp-getting-started/test.csv")

tr.head()

ts.head()
d=tr.keyword.isnull().value_counts()

d

plt.pie(d,labels=(True,False))

plt.legend()

plt.show()
b=ts.location.isnull().value_counts()

b

from matplotlib import style

style.use('seaborn-deep')

plt.bar(b.index,b.values)

plt.title('Location',size=20)

plt.show()
c=tr.target.value_counts()

c

style.use('classic')

plt.bar(c.index,c.values)

plt.title('Target',size=20)

plt.show()
xtrain=tr.loc[:,['text']]

ytrain=tr.loc[:,['target']]

xtest=ts.loc[:,['text']]

xtest
from nltk.corpus import stopwords

sw=stopwords.words('english')

print(sw)

import string

def text_cleaning(a):

    remove_punctuation=[char for char in a if char not in string.punctuation]

    remove_punctuation=''.join(remove_punctuation)

    return ["".join(word) for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]

z=[]

for j in xtrain,xtest:

    #print(text_cleaning(j))

    text=" ".join(text_cleaning(j))

    z.append(text)

z

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

a=cv.fit_transform(xtrain.iloc[:,0])

b=cv.transform(xtest.iloc[:,0])

print(b)
#x=a.toarray()

#y=b.toarray()

#x

#y
from sklearn.naive_bayes import BernoulliNB

bn=BernoulliNB()

bn.fit(a,ytrain)

pred=bn.predict(b)

pred



from sklearn.naive_bayes import MultinomialNB

mb=MultinomialNB()

mb.fit(a,ytrain)

pred1=mb.predict(b)

pred1
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(a,ytrain)

pred2=lr.predict(b)

pred2


plt.plot(pred)

plt.show()