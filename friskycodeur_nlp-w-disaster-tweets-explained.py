import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import re
import string
import nltk
from nltk.corpus import stopwords
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from wordcloud import WordCloud,STOPWORDS
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')
dataset=pd.concat([train,test])
print(f'train:{train.shape}\ntest:{test.shape}\ndataset:{dataset.shape}')
train.head()
test.head()
(train.isnull().sum()[train.isnull().sum()>0]/len(train))*100
pd.DataFrame({'Test Data Missing':(test.isnull().mean()*100).sort_values(ascending=False)})
non_dis = train[train.target==0]['text']
non_dis.values[7]
dis=train[train.target==1]['text']
dis.values[7]
train.target.value_counts()
plt.figure(figsize=(6,6))
sns.barplot(train.target.value_counts().index,train.target.value_counts())
train.keyword.nunique()
plt.figure(figsize=(12,12))
sns.barplot(y=train.keyword.value_counts().index[:15],x=train.keyword.value_counts()[:15])
print(train.location.nunique())
plt.figure(figsize=(12,12))
sns.barplot(y=train.location.value_counts().index[:15],x=train.location.value_counts()[:15])
plt.figure(figsize=(12,12))
sns.barplot(y=train.location.value_counts().index[-10:],x=train.location.value_counts()[-10:])
def lowercase_text(text):
    return text.lower()

train.text=train.text.apply(lambda x: lowercase_text(x))
test.text=test.text.apply(lambda x: lowercase_text(x))
train.text.head(5)
def remove_noise(text):
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
train.text=train.text.apply(lambda x: remove_noise(x))
test.text=test.text.apply(lambda x: remove_noise(x))
train.text.head(5)
!pip install nlppreprocess
from nlppreprocess import NLP

nlp = NLP()

train['text'] = train['text'].apply(nlp.process)
test['text'] = test['text'].apply(nlp.process)  
train.text.sample(10)
stemmer = SnowballStemmer("english")

def stemming(text):
    text = [stemmer.stem(word) for word in text.split()]
    return ' '.join(text)

train['text'] = train['text'].apply(stemming)
test['text'] = test['text'].apply(stemming)
from wordcloud import WordCloud
fig , ax1 = plt.subplots(1,figsize=(12,12))
wordcloud=WordCloud(background_color='white',width=600,height=600).generate(" ".join(train.text))
ax1.imshow(wordcloud)
ax1.axis('off')
ax1.set_title('Frequent Words',fontsize=24)
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer=CountVectorizer(analyzer='word',binary=True)
count_vectorizer.fit(train.text)

train_vec = count_vectorizer.fit_transform(train.text)
test_vec = count_vectorizer.transform(test.text)

print(train_vec[7].todense())
print(test_vec[7].todense())
y=train.target
from sklearn import model_selection
model =MultinomialNB(alpha=1)
scores= model_selection.cross_val_score(model,train_vec,y,cv=6,scoring='f1')
scores
model.fit(train_vec,y)
sample_submission=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
sample_submission.target= model.predict(test_vec)
sample_submission.head()
sample_submission.to_csv('submission.csv',index=False)