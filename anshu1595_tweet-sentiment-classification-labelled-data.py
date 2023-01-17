# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/train_data.csv')
data.head()
data.isnull().sum()
data=data[['sentiment','text']]
data.head()
data.isnull().sum()
data=data.dropna()
data.isnull().sum()
y=data['sentiment']
x=data['text']
x.head()
y.head()
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import spacy
from spacy import displacy
import string
import re
punct = string.punctuation
punct
nlp = spacy.load('en_core_web_sm')
len(punct)
def text_cleaning(sen):
    sen = re.sub("http?\S+","",sen)
    sen = sen.translate(str.maketrans(punct,32*" "))
    doc = nlp(sen)
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_.strip()
        tokens.append(temp)
        
    tokens = [token for token in tokens if token not in punct and token not in stopwords.words('english')]
    return tokens
tt = TweetTokenizer()
text_cleaning("#BREAKING: Lockdown in containment zones extended till June-30,2020")
tt.tokenize("If you don't want to talk to me tell me instead of leading me on and making excuses just tell me then fuck off-C")
# data['text'] = data['text'].astype(str)
# data['text'] = data['text'].apply(str.lower)
# cleaned_text = []
# for row in data.iterrows():
#     tokens = tknzr.tokenize(row[1][0])
#     tokens = [t.translate(str.maketrans("","",punctuations)) for t in tokens]
#     tokens = [t for t in tokens if not t.startswith('@')]
#     cleaned_text.append(' '.join(tokens))
# data['cleaned_text'] = pd.Series(cleaned_text)
# data['cleaned_text'] = data['cleaned_text'].apply(lambda x: re.sub("http?\S+","",x)).apply(str.strip)
# data_cleaned = data['cleaned_text']
# data_cleaned.head(10)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=1111)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
tfidf = TfidfVectorizer(tokenizer=text_cleaning, ngram_range=(1,3),max_df=0.6, max_features=15000)
clf_mnb = MultinomialNB(alpha=0.5)
clf_svc = SVC(probability=True)
pipe_mnb = Pipeline([('tfidf',tfidf),('clf_mnb',clf_mnb)], verbose=50)
pipe_svc = Pipeline([('tfidf',tfidf),('clf_svc',clf_svc)], verbose=50)
pipe_mnb.fit(X_train,y_train)
print(pipe_mnb.score(X_train,y_train)*100)
print(pipe_mnb.score(X_test,y_test)*100)
pipe_svc.fit(X_train,y_train)
print(pipe_svc.score(X_train,y_train)*100)
print(pipe_svc.score(X_test,y_test)*100)
pipe_mnb[0].vocabulary_
y_pred_mnb = pipe_mnb.predict(X_test)
y_pred_svc = pipe_svc.predict(X_test)
print(confusion_matrix(y_test,y_pred_mnb))
print(classification_report(y_test,y_pred_mnb))
print(confusion_matrix(y_test,y_pred_svc))
print(classification_report(y_test,y_pred_svc))
pipe_mnb.predict_proba(["I am very happy with my results"])
pipe_svc.predict_proba(["I am very happy with my results"])
pipe_mnb.predict(["I am very happy with my results"])
pipe_svc.predict(["I am very happy with my results"])