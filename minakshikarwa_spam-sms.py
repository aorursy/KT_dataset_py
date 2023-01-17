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
import re



import matplotlib.pyplot as plt 

import seaborn as sns

from string import punctuation

import warnings 

warnings.filterwarnings("ignore", category=DeprecationWarning)

from wordcloud import WordCloud, STOPWORDS



%matplotlib inline



from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split





from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import *

from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import MultinomialNB
sms =  pd.read_csv('/kaggle/input/sms-spam/spamraw.csv')
sms.head(10)
sms.type.value_counts()
together = "".join(sms['text'])

ham_sms = " ".join(sms.loc[sms['type']=="ham",'text'])

spam_sms = " ".join(sms.loc[sms['type']=="spam",'text'])

#together
wordcloud = WordCloud().generate(together)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud().generate(ham_sms)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', 

                      width=4000,height=2000).generate(spam_sms)



plt.imshow(wordcloud)

plt.axis('off')

plt.show()
stop = set(stopwords.words('english')+list('punctuation'))

len(stop)
lemma = WordNetLemmatizer()

ps = PorterStemmer()
def split_into_lemmas(message):

    message = message.lower()

    # remove special characters, numbers, punctuations

    message =re.sub("[^a-zA-Z ]+", " ", message)

    #Removing Short Words

    #message = message.apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

    #tokenization, separating words from sentences

    message = word_tokenize(message)

    #stemming : Stemming is a rule-based process of stripping the suffixes (“ing”, “ly”, “es”, “s” etc) from a word

    words = [ps.stem(m) for m in message]

    #[stemmer.stem(plural) for plural in plurals]

    words_sans_stop=[]

    for word in words :

        if word in stop:continue

        words_sans_stop.append(word)

    return [lemma.lemmatize(word) for word in words_sans_stop]

    
Y = sms['type']

X = sms['text']

X.head()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
tfidf_vectorizer = TfidfVectorizer(analyzer=split_into_lemmas, max_df = 0.9, min_df = 5, max_features = 500, stop_words=stop)

# TF-IDF feature matrix

tfidf = tfidf_vectorizer.fit(x_train)
tfidf.get_feature_names()
train1 = tfidf.transform(x_train)

test1 = tfidf.transform(x_test)

train1.shape, test1.shape
clf= MultinomialNB()

clf.fit(train1, y_train)
predictions=pd.DataFrame(list(zip(y_test,clf.predict(test1))),columns=['real','predicted'])



pd.crosstab(predictions['real'],predictions['predicted'])
from sklearn.metrics import accuracy_score
pred =clf.predict(test1)

pred
accuracy_score(y_test,pred )