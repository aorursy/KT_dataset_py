import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import nltk

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/train.csv')

df.head()
docs = df['question_text']

len(docs)
df_0 = df[df['target']==0]

docs_0 = df_0['question_text']

df_1 = df[df['target']==1]

docs_1 = df_1['question_text']
!pip install wordcloud
from wordcloud import WordCloud

stopwords = nltk.corpus.stopwords.words('english')

wc = WordCloud(background_color = 'white',stopwords=stopwords).generate(' '.join(docs_0))

plt.figure(figsize=(16,8))

plt.imshow(wc)
from wordcloud import WordCloud

stopwords = nltk.corpus.stopwords.words('english')

wc = WordCloud(background_color = 'white',stopwords=stopwords).generate(' '.join(docs_1))

plt.figure(figsize=(16,8))

plt.imshow(wc)
docs = docs.str.lower().str.replace('[^a-z ]', '')

docs.head()

docs.shape
stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.PorterStemmer()



def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean = ' '.join(words_clean)

    return doc_clean



docs_clean = docs.apply(clean_sentence)

docs_clean.head()
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer().fit(docs_clean)

dtm = vectorizer.transform(docs_clean)

dtm
non_zero = 8041045

zeors_count = (1306122 * 178080) - non_zero

sparsity = zeors_count/(1306122 * 178080) * 100

sparsity
docs_clean.shape

df['target'].shape
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(docs_clean,

                                                 df['target'],

                                                 test_size = 0.2,

                                                 random_state = 100)

vectorizer = CountVectorizer(min_df=50).fit(train_x)

train_x = vectorizer.transform(train_x)

test_x = vectorizer.transform(test_x)

train_x.shape,test_x.shape
from sklearn.naive_bayes import MultinomialNB

model_mnb1 = MultinomialNB().fit(train_x,train_y)

test_pred = model_mnb1.predict(test_x)
from sklearn.metrics import classification_report,f1_score
f1_score(test_y,test_pred)
df_test = pd.read_csv('../input/test.csv')
docs = df_test['question_text']
docs = docs.str.lower().str.replace('[^a-z ]', '')

docs.head()
stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.PorterStemmer()



def clean_sentence(doc):

    words = doc.split(' ')

    words_clean = [stemmer.stem(word) for word in words if word not in stopwords]

    doc_clean = ' '.join(words_clean)

    return doc_clean



docs_clean = docs.apply(clean_sentence)

docs_clean.head()
docs_clean = vectorizer.transform(docs_clean)
final = model_mnb1.predict(docs_clean)

final = pd.DataFrame({'qid':df_test['qid'],'prediction':final})
final.to_csv('submission.csv',index=False)