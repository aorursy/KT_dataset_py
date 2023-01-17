import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')
df=pd.read_json('../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json',lines=True)
df.head()
df.info()
#how many headlines are sarcastic, how many aren't
plt.figure(figsize=(12,3))
plt.subplot(1,2,1)
sns.countplot(df['is_sarcastic'])
plt.title('class value_counts');
plt.subplot(1,2,2)
df['is_sarcastic'].value_counts().plot(kind='pie', autopct='%1.1f%%');
plt.title('class proportions');
#manual train_test_split for realism 
#('new' data will be fitted to the already existing CountVectorizer representation)

rnd=np.random.RandomState(33)
tsin=rnd.choice(np.arange(df.shape[0]), df.shape[0]//10,replace=False)

x_tr=df.iloc[np.delete(np.arange(df.shape[0]),tsin)]['headline'].values
y_tr=df.iloc[np.delete(np.arange(df.shape[0]),tsin)]['is_sarcastic'].values
x_ts=df.iloc[tsin]['headline'].values
y_ts=df.iloc[tsin]['is_sarcastic'].values
#count vectorizer with 1-gram
count_vect=CountVectorizer()
x_tr_cv=count_vect.fit_transform(x_tr)
x_ts_cv=count_vect.transform(x_ts)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_cv,y_tr)
y_pred=lr.predict(x_ts_cv)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
#count vectorizer with 1-gram and 2-gram
count_vect2=CountVectorizer(ngram_range=(1,2))
x_tr_cv2=count_vect2.fit_transform(x_tr)
x_ts_cv2=count_vect2.transform(x_ts)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_cv2,y_tr)
y_pred=lr.predict(x_ts_cv2)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
#count vectorizer with 1-gram,2-gram, and 3-gram
count_vect3=CountVectorizer(ngram_range=(1,3))
x_tr_cv3=count_vect3.fit_transform(x_tr)
x_ts_cv3=count_vect3.transform(x_ts)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_cv3,y_tr)
y_pred=lr.predict(x_ts_cv3)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
#tfidf vectorizer
tfidf=TfidfVectorizer()
x_tr_tfidf=tfidf.fit_transform(x_tr)
x_ts_tfidf=tfidf.transform(x_ts)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_tfidf,y_tr)
y_pred=lr.predict(x_ts_tfidf)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
#tfidf vectorizer with 1-gram and 2-gram
tfidf2=TfidfVectorizer(ngram_range=(1,2))
x_tr_tfidf2=tfidf2.fit_transform(x_tr)
x_ts_tfidf2=tfidf2.transform(x_ts)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_tfidf2,y_tr)
y_pred=lr.predict(x_ts_tfidf2)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
#count vectorizer, binary encoding, with 1-gram and 2-gram
count_vect_bin=CountVectorizer(ngram_range=(1,2), binary=True)
x_tr_bin=count_vect_bin.fit_transform(x_tr)
x_ts_bin=count_vect_bin.transform(x_ts)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_bin,y_tr)
y_pred=lr.predict(x_ts_bin)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
excl=0
ind=[]
for line,se in enumerate(x_tr):
    for c in se: 
        if c=='!':
            excl=excl+1
            ind.append(line)

print('number of exclamation marks in dataset:',excl)
print('number of sarcastic headlines containing exclamation marks:',y_tr[ind].cumsum()[-1])
print('number of non-sarcastic headlines containing exclamation marks:', excl-(y_tr[ind].cumsum()[-1]))

#decapitalize and remove non-alphanumericals
def clean(text_array):
    clean_text=[]
    for i,line in enumerate(text_array):
        clean_text.append(re.sub('[\W]+', ' ',re.sub('\'', '',line.lower())))
    return clean_text

x_tr_p=clean(x_tr)
x_ts_p=clean(x_ts)

#sanity check
x_ts_p[:2]
#count vectorizer, binary encoding, with 1-gram and 2-gram, clean text
count_vect_bin2=CountVectorizer(ngram_range=(1,2), binary=True)
x_tr_p_bin=count_vect_bin2.fit_transform(x_tr_p)
x_ts_p_bin=count_vect_bin2.transform(x_ts_p)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_p_bin,y_tr)
y_pred=lr.predict(x_ts_p_bin)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
#count vectorizer, binary encoding, with 1-gram and 2-gram, clean text, removed stopwords
vect=CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')
x_tr_stop=vect.fit_transform(x_tr_p)
x_ts_stop=vect.transform(x_ts_p)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_stop,y_tr)
y_pred=lr.predict(x_ts_stop)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))
#count vectorizer, binary encoding, with 1-gram and 2-gram, raw text, removed stopwords
vect2=CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')
x_tr_stop2=vect2.fit_transform(x_tr)
x_ts_stop2=vect2.transform(x_ts)

lr=LogisticRegression(random_state=7)
lr.fit(x_tr_stop2,y_tr)
y_pred=lr.predict(x_ts_stop2)

print('accuracy:', accuracy_score(y_ts,y_pred))
print('f1-score:', f1_score(y_ts,y_pred))