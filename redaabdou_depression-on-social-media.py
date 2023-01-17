import re    # for regular expressions 

import nltk  # for text manipulation 

import string # for text manipulation 

import warnings 

import numpy as np 

import pandas as pd # for data manipulation 

import matplotlib.pyplot as plt



pd.set_option("display.max_colwidth", 200) 

warnings.filterwarnings("ignore") #ignore warnings



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
data = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv",encoding='latin-1')

data.head()
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]

data.columns = DATASET_COLUMNS

data.head()
data.drop(['ids','date','flag','user'],axis = 1,inplace = True)
data.head()
positif_data = data[data.target==4].iloc[:25000,:]

print(positif_data.shape)

negative_data = data[data.target==0].iloc[:1000,:]

print(negative_data.shape)
data = pd.concat([positif_data,negative_data],axis = 0)

print(data.shape)

data.head()
data['Clean_TweetText'] = data['TweetText'].str.replace("@", "") 

data.head()
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace(r"http\S+", "") 

data.head()
data['Clean_TweetText'] = data['Clean_TweetText'].str.replace("[^a-zA-Z]", " ") 

data.head()
stopwords=nltk.corpus.stopwords.words('english')
def remove_stopwords(text):

    clean_text=' '.join([word for word in text.split() if word not in stopwords])

    return clean_text
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda text : remove_stopwords(text.lower()))

data.head()
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: x.split())

data.head()
from nltk.stem.porter import * 

stemmer = PorterStemmer() 

data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: [stemmer.stem(i) for i in x])

data.head()
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x]))

data.head()
data['Clean_TweetText'] = data['Clean_TweetText'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

data.head()
all_words = ' '.join([text for text in data['Clean_TweetText']])



from wordcloud import WordCloud 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words) 



plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
positive_words =' '.join([text for text in data['Clean_TweetText'][data['target'] == 4]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)



plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
depressive_words =' '.join([text for text in data['Clean_TweetText'][data['target'] == 0]]) 

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(depressive_words)



plt.figure(figsize=(10, 7)) 

plt.imshow(wordcloud, interpolation="bilinear") 

plt.axis('off') 

plt.show()
from xgboost import XGBClassifier

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier 

from sklearn.linear_model import LogisticRegression 

from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english') 

cv = count_vectorizer.fit_transform(data['Clean_TweetText'])

cv.shape
X_train,X_test,y_train,y_test = train_test_split(cv,data['target'] , test_size=.2,stratify=data['target'], random_state=42)
xgbc = XGBClassifier(max_depth=6, n_estimators=1000, nthread= 3)

xgbc.fit(X_train,y_train)

prediction_xgb = xgbc.predict(X_test)

print(accuracy_score(prediction_xgb,y_test))
rf = RandomForestClassifier(n_estimators=1000, random_state=42)

rf.fit(X_train,y_train)

prediction_rf = rf.predict(X_test)

print(accuracy_score(prediction_rf,y_test))
lr = LogisticRegression()

lr.fit(X_train,y_train)

prediction_lr = lr.predict(X_test)

print(accuracy_score(prediction_lr,y_test))
svc = svm.SVC()

svc.fit(X_train,y_train)

prediction_svc = svc.predict(X_test)

print(accuracy_score(prediction_svc,y_test))