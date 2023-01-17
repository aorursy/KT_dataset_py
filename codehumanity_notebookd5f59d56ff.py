import matplotlib.pyplot as plt

import nltk

import pandas as pd

import numpy as np

import seaborn as sns

import string

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

%matplotlib inline
# This is for setting the theme of the graph

sns.set(style="darkgrid")
combine_news = pd.read_csv('../input/Combined_News_DJIA.csv')
# First lets see how our data look

combine_news.head()
combine_news.info()
sns.countplot(x='Label',data=combine_news)
def news_process(news):

    '''

    this function takes news as an argument,then remove all the punctuation marks and 

    all the stop words in the news, finally return a list of words.

    '''

    if isinstance(news,str):

        news = news.strip('b')

        news = [c for c in news if c not in string.punctuation]

        news = ''.join(news)

        return news

    else:

        return ''
headlines = []

for row in range(0,len(combine_news.index)):

    headlines.append(' '.join(news_process(news) for news in combine_news.iloc[row,2:27]))

combine_news['headlines'] = headlines
df = combine_news[['Date','Label','headlines']]

df.head()
def clean_news(news):

    stemmer = SnowballStemmer("english", ignore_stopwords=True)    

    clean_news = [stemmer.stem(word) for word in news.split() if word.lower() not in stopwords.words('english')]

    return clean_news
from sklearn.feature_extraction.text import CountVectorizer
# Might take awhile...

bow_transformer = CountVectorizer(analyzer=clean_news).fit(df['headlines'])



# Print total number of vocab words

print(len(bow_transformer.vocabulary_))
headline4 = df['headlines'][3]

print(headline4)
bow4 = bow_transformer.transform([headline4])

print(bow4)

print(bow4.max())
print(bow_transformer.get_feature_names()[26693])
headlines_bow = bow_transformer.transform(df['headlines'])
print('Shape of Sparse Matrix: ', headlines_bow.shape)

print('Amount of Non-Zero occurences: ', headlines_bow.nnz)
sparsity = (100.0 * headlines_bow.nnz / (headlines_bow.shape[0] * headlines_bow.shape[1]))

print('sparsity: {}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(headlines_bow)

tfidf4 = tfidf_transformer.transform(bow4)

print(tfidf4)
headlines_tfidf = tfidf_transformer.transform(headlines_bow)

print(headlines_tfidf.shape)
headline_train = df[df['Date']<'2015-01-01']['headlines']

label_train = df[df['Date']<'2015-01-01']['Label']

headline_test = df[df['Date']>'2014-12-31']['headlines']

label_test = df[df['Date']>'2014-12-31']['Label']
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=clean_news)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
pipeline.fit(headline_train,label_train)
predictions = pipeline.predict(headline_test)
from sklearn.metrics import classification_report

print(classification_report(label_test,predictions))
from sklearn import metrics 

metrics.accuracy_score(predictions,label_test)
y_pred_prob = pipeline.predict_proba(headline_test)[:,1]
metrics.roc_auc_score(label_test,y_pred_prob) 
df_one_day_before = df[:]

df_one_day_before['Label'] = df_one_day_before['Label'].shift(-1)

df_yesterday = df_one_day_before[:-1]

df_yesterday
headline_train = df_yesterday[df_yesterday['Date']<'2015-01-01']['headlines']

label_train = df_yesterday[df_yesterday['Date']<'2015-01-01']['Label']

headline_test = df_yesterday[df_yesterday['Date']>'2014-12-31']['headlines']

label_test = df_yesterday[df_yesterday['Date']>'2014-12-31']['Label']
pipeline.fit(headline_train,label_train)
predictions = pipeline.predict(headline_test)
from sklearn.metrics import classification_report

print(classification_report(label_test,predictions))
df_one_day_before = df[:]

df_one_day_before['Label'] = df_one_day_before['Label'].shift(-1)

df_yesterday = df_one_day_before[:-1]

df_yesterday

headline_train = df_yesterday[df_yesterday['Date']<'2015-01-01']['headlines']

label_train = df_yesterday[df_yesterday['Date']<'2015-01-01']['Label']

headline_test = df_yesterday[df_yesterday['Date']>'2014-12-31']['headlines']

label_test = df_yesterday[df_yesterday['Date']>'2014-12-31']['Label']
from sklearn.naive_bayes import BernoulliNB

bernoulli_pipeline = Pipeline([

    ('bow', CountVectorizer(ngram_range=(1,2),analyzer=clean_news)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', BernoulliNB(alpha=0.5,binarize=0.0)),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
bernoulli_pipeline.fit(headline_train,label_train)
predictions = bernoulli_pipeline.predict(headline_test)
print(classification_report(label_test,predictions))
metrics.accuracy_score(label_test,predictions)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(label_test,predictions))
metrics.accuracy_score(predictions,label_test)
y_pred_prob = bernoulli_pipeline.predict_proba(headline_test)[:,1]
metrics.roc_auc_score(label_test,y_pred_prob) 