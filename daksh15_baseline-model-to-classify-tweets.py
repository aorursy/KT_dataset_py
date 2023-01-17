from sklearn.feature_extraction.text import TfidfVectorizer 

import pandas as pd

import numpy as np

import spacy

import seaborn as sns

import matplotlib.pyplot as plt

import nltk

from nltk.tokenize import word_tokenize

from sklearn.model_selection import cross_val_score

import re

from sklearn .metrics import classification_report

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import string

from nltk.tokenize import WhitespaceTokenizer,word_tokenize
#reading train and test data

train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
print(f"training_data_shape:{train.shape}")

print(f"testing_data_shape:{test.shape}")
train.head(10)
#function to get percentage of Nan in a Column

def get_nan(data):

    s=data.isna().sum()

    per=s/data.shape[0]

    return per*100
train_nan=get_nan(train)

test_nan=get_nan(test)

nan_data=pd.DataFrame({"train":train_nan,"test":test_nan})

nan_data
train.info()
sns.countplot(x='target',data=train,palette='Blues')

sns.barplot(y=train['keyword'].value_counts()[:20].index,x=train['keyword'].value_counts()[:20])
train['location'].replace({'United States':'USA',

                           'New York':'USA',

                            "London":'UK',

                            "Los Angeles, CA":'USA',

                            "Washington, D.C.":'USA',

                            "California":'USA',

                             "Chicago, IL":'USA',

                             "Chicago":'USA',

                            "New York, NY":'USA',

                            "California, USA":'USA',

                            "FLorida":'USA',

                            "Nigeria":'Africa',

                            "Kenya":'Africa',

                            "Everywhere":'Worldwide',

                            "San Francisco":'USA',

                            "Florida":'USA',

                            "United Kingdom":'UK',

                            "Los Angeles":'USA',

                            "Toronto":'Canada',

                            "San Francisco, CA":'USA',

                            "NYC":'USA',

                            "Seattle":'USA',

                            "Earth":'Worldwide',

                            "Ireland":'UK',

                            "London, England":'UK',

                            "New York City":'USA',

                            "Texas":'USA',

                            "London, UK":'UK',

                            "Atlanta, GA":'USA',

                            "Mumbai":"India"},inplace=True)



sns.barplot(y=train['location'].value_counts()[:5].index,x=train['location'].value_counts()[:5],

            orient='h')
#function to clean text

def clean_text(text):

    text=text.lower()

    text = re.sub('\[.*?\]', "", text)

    text = re.sub('https?://\S+|www\.\S+', "", text)

    text=re.sub('<.*?>+','',text)

    text=re.sub(f"[{string.punctuation}]",'',text)

    text=re.sub('\n','',text)

    text=re.sub('\w\d\w','',text)

    return text

   

    
#applying the cleaning process on both train and test

train['text']=train['text'].apply(lambda x:clean_text(x))

test['text']=train['text'].apply(lambda x:clean_text(x))

disaster_tweet=train[train['target']==1]['text']

non_disaster_tweet=train[train['target']==0]['text']
from wordcloud import WordCloud

fig,(ax1,ax2)=plt.subplots(1,2,figsize=[28,7])

wordcloud1=WordCloud(background_color='white',height=100,width=200).generate(" ".join(disaster_tweet))

ax1.imshow(wordcloud1)

ax1.axis('off')

ax1.set_title('Disaster Tweets',fontsize=32)

wordcloud2=WordCloud(background_color='white',height=100,width=200).generate(" ".join(non_disaster_tweet))

ax2.imshow(wordcloud2)

ax2.axis('off')

ax2.set_title('Non-Disaster Tweets',fontsize=32)
#function to remove stopwords

def stop_words_removal(text):

    text=[w for w in  text.split(" ")  if w not in stopwords.words('english')]

    return " ".join(text)

#applying the function on train and test

train['text']=train['text'].apply(lambda x: stop_words_removal(x))

test['text']=test['text'].apply(lambda x:stop_words_removal(x))
#looking at top three rows after cleaning

train['text'][0:3]
tf_idf=TfidfVectorizer(max_features=1000)

train_X=tf_idf.fit_transform(train['text'])

test_X=tf_idf.transform(test['text'])
from sklearn.linear_model import LogisticRegression

train_Y=train['target']

log_reg=LogisticRegression(C=2.02)

scores=cross_val_score(log_reg,train_X,train_Y,cv=5,scoring='f1')

print(scores)
log_reg.fit(train_X,train_Y)
from sklearn.naive_bayes import MultinomialNB

Bayes=MultinomialNB(alpha=1.5)

scores_bayes=cross_val_score(Bayes,train_X,train_Y,cv=5,scoring='f1')

print(scores_bayes)
Bayes.fit(train_X,train_Y)
from   xgboost import XGBClassifier as xgb

xg=xgb(n_estimators=250,max_depth=12,colsample_bytree=0.1,learning_rate=0.2,subsample=0.4)

xg_scores=cross_val_score(xg,train_X,train_Y,cv=5,scoring='f1')

print(xg_scores)
def submit_model(path,model,vectors):

    sample_submission=pd.read_csv(path)

    sample_submission['target']=model.predict(vectors)

    sample_submission.to_csv('submission.csv',index=False)

submission_file_path = "../input/nlp-getting-started/sample_submission.csv"

test_vectors=test_X

submit_model(submission_file_path,Bayes,test_vectors)