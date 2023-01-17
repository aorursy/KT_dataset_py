# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import make_scorer,f1_score, accuracy_score, precision_score

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dfTickets = pd.read_csv('../input/all_tickets.csv')

dfTickets.info()
dfTickets.head()
sns.heatmap(dfTickets.isnull())
dfTickets.isnull().sum()
newdf = dfTickets.select_dtypes([np.number])

newdf.columns.values

dfTickets['ticket_type'].value_counts()
dfTickets['category'].value_counts()
dfTickets['sub_category1'].value_counts()
len(dfTickets['sub_category1'].value_counts())
dfTickets['sub_category2'].value_counts()
len(dfTickets['sub_category2'].value_counts())
dfTickets['business_service'].value_counts()

dfTickets['urgency'].value_counts()

dfTickets['impact'].value_counts()

dfTickets['title'].isna().sum()
dfTickets['body'].isna().sum()
dfTickets.shape
dfTickets.shape
Y= pd.DataFrame(dfTickets['ticket_type'])
X = dfTickets.drop(columns=["title","ticket_type"])
print(type(Y))

print(type(X))
dfTickets.head()
X_train, X_test, y_train, y_test = train_test_split(

    X['body'], Y, test_size=0.4, random_state=0)

print(X_train.shape)

print(y_train.shape)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_tf = cv.fit_transform(X_train)

X_train_tf.shape
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf= tfidf_transformer.fit_transform(X_train_tf)

X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score, accuracy_score

mnb = MultinomialNB()

mnb.fit(X_train_tfidf, y_train)
X_test_tf = cv.transform(X_test)

X_test_tfidf = tfidf_transformer.transform(X_test_tf)

pred = mnb.predict(X_test_tfidf)

print(round(f1_score(y_test, pred),2))

print(round(accuracy_score(y_test, pred),2))

print('***Stemming***')



from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):

        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

X_train_counts = stemmed_count_vect.fit_transform(X_train)





from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()

X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape



from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score, accuracy_score

mnb = MultinomialNB()

mnb.fit(X_train_tfidf, y_train)

X_test_tf = stemmed_count_vect.transform(X_test)

X_test_tfidf = tfidf_transformer.transform(X_test_tf)

pred = mnb.predict(X_test_tfidf)

print('MultinomialNB')

print(round(f1_score(y_test, pred),2))

print(round(accuracy_score(y_test, pred),2))