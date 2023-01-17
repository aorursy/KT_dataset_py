import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
train.drop(['keyword','location'],axis=1,inplace=True)

test.drop(['keyword','location'],axis=1,inplace=True)
train.head()
test.head()
import string
train['text']=train['text'].str.lower()

test['text']=test['text'].str.lower()
text=train['text']

text1=test['text']
def remove_punctuation(text):

    return text.translate(str.maketrans('','',string.punctuation))

text_clean=text.apply(lambda text:remove_punctuation(text))

text_clean1=text1.apply(lambda text1:remove_punctuation(text1))
text_clean.head()
text_clean1.head()
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
def stopwords_(text):

    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

text_clean = text_clean.apply(lambda text: stopwords_(text))

text_clean1 = text_clean1.apply(lambda text1: stopwords_(text1))
text_clean.head()
text_clean1.head()
from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

def lemma(text):

    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
import nltk

from nltk.stem import WordNetLemmatizer   

lemmatizer = WordNetLemmatizer() 

text_clean=text_clean.apply(lambda text: lemma(text))

text_clean1=text_clean1.apply(lambda text1: lemma(text1))
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
import re

text_clean=text_clean.apply(lambda x : remove_URL(x))

text_clean1=text_clean1.apply(lambda x : remove_URL(x))
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)
text_clean=text_clean.apply(lambda x : remove_html(x))

text_clean1=text_clean1.apply(lambda x : remove_html(x))
text_clean.head()
text_clean1.head()
from wordcloud import WordCloud
all_words = ' '.join([text for text in text_clean])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(16, 10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.show()

df = pd.DataFrame({"text": text_clean})

df.head()
train.update(df)
train.head()
df1 = pd.DataFrame({"text": text_clean1})

df1.head()
test.update(df1)
test.drop('id',axis=1,inplace=True)
test.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_all = pd.concat([train["text"],test["text"]])



tfidf = TfidfVectorizer(stop_words = 'english')

tfidf.fit(X_all)



X = tfidf.transform(train["text"])

X_test = tfidf.transform(test["text"])

del X_all
x=X

y=train.iloc[:,-1]
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit = LogisticRegression(penalty='l2',solver='saga',l1_ratio=0.2)
logit.fit(x,y)
x1=X_test
y1=logit.predict(x1)
sample_submission.head()
from sklearn.metrics import accuracy_score
score =accuracy_score(sample_submission['target'],y1)
score*100
from sklearn.linear_model import SGDClassifier
clf=SGDClassifier(loss='modified_huber',verbose=1)
clf.fit(x,y)
y2=clf.predict(x1)
from sklearn.metrics import accuracy_score
score1 =accuracy_score(sample_submission['target'],y1)
score1
from sklearn.model_selection import GridSearchCV

from sklearn import svm
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000], 'gamma' : [0.001,0.0001]}
clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameters, n_jobs=-1)
clf.fit(x,y)
y3=clf.predict(x1)
score2 =accuracy_score(sample_submission['target'],y3)
score2
prediction = pd.DataFrame(y3, columns=['y3']).to_csv('prediction.csv')