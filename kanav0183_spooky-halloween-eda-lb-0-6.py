import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from wordcloud import WordCloud, STOPWORDS

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
print(train.id[0],'---')

print(train.text[0])
plt.figure(figsize=(10,6))

sns.countplot(train['author']);

plt.title('Countplot for Authours_');

plt.xlabel('Authors_',fontsize=20);
df_eap = train[train.author=='EAP']

df_hpl = train[train.author=='HPL']

df_mws = train[train.author=='MWS']
df_eap.text

dic= (' '.join(df_eap['text']))



wordcloud = WordCloud(width = 1000, height = 500,stopwords=STOPWORDS).generate(dic)



plt.figure(figsize=(15,5));

plt.imshow(wordcloud);

plt.axis('off');

plt.title('Word Cloud for EAP');
dic= (' '.join(df_hpl['text']))



wordcloud = WordCloud(width = 1000, height = 500,stopwords=STOPWORDS).generate(dic)



plt.figure(figsize=(15,5));

plt.imshow(wordcloud);

plt.axis('off');

plt.title('Word Cloud for HPL');


dic= (' '.join(df_mws['text']))



wordcloud = WordCloud(width = 1000, height = 500,stopwords=STOPWORDS).generate(dic)



plt.figure(figsize=(15,5));

plt.imshow(wordcloud);

plt.axis('off');

plt.title('Word Cloud for MWS');
train["length"] = train["text"].apply(lambda x: len(str(x).split()))

test["length"] = test["text"].apply(lambda x: len(str(x).split()))

print(train['length'].head())
train["stp_len"] = train["text"].apply(lambda x: len([i for i in str(x).lower().split() if i in stopwords.words("english")]))

test["stp_len"] = test["text"].apply(lambda x: len([i for i in str(x).lower().split() if i in stopwords.words("english")]))

print(train['stp_len'].head())
print(train.groupby(by=['author'])['length'].mean())

train.groupby(by=['author'])['length'].mean().plot(kind='bar');
print(train.groupby(by=['author'])['stp_len'].mean())

train.groupby(by=['author'])['stp_len'].mean().plot(kind='bar');
import string

train["punct"] =train['text'].apply(lambda x: len([i for i in str(x) if i in string.punctuation]) )

test["punct"] =test['text'].apply(lambda x: len([i for i in str(x) if i in string.punctuation]) )
import matplotlib.pyplot as plt

%matplotlib inline

print(train.groupby(by=['author'])['punct'].mean())

train.groupby(by=['author'])['punct'].mean().plot(kind='bar');
vect = CountVectorizer(ngram_range=(1,2),min_df=5).fit(train['text'])

train_vectorized = vect.transform(train['text'])

len(vect.get_feature_names())
k=['length', 'stp_len', 'punct']

arr = np.array(train[k])

import scipy

stack = train[['length', 'stp_len', 'punct']]

train_vectorized = scipy.sparse.hstack([train_vectorized, stack])
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=.03,n_jobs=-1)

model.fit(train_vectorized, train['author'])
k=np.array(test[['length', 'stp_len', 'punct']])

X_test = vect.transform(test['text'] )

X_test = scipy.sparse.hstack([X_test, k])
df = pd.DataFrame(model.predict_proba(X_test),index=test['id'],columns=['EAP','HPL','MWS'])

df.head()
df.to_csv('Submission1.csv')