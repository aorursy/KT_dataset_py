import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as TSVD
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
data = pd.read_csv('../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')
data.head()
data.shape
train = data.iloc[:25000]
test = data.iloc[25000:]
train.shape, test.shape
train.head()
nlp = spacy.load('en_core_web_lg')
# Converting the text to lowercase

train['review'] = train['review'].apply(lambda x: str(x).lower())
data.head()
!pip install contractions
import contractions
contractions_dict = contractions.contractions_dict
contractions_dict
def contraction_expansion(x):
    
    if type(x) is str:
        
        for key in contractions_dict:
            
            value = contractions_dict[key]
            
            x = x.replace(key, value)
            
        return x
    
    else:
        
        return x
train['review'] = train['review'].apply(lambda x: contraction_expansion(x))
train.head()
def remove_emails(x):
    
    email_pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    
    return re.sub(email_pattern, '', x)
train['review'] = train['review'].apply(lambda x:remove_emails(x))
train.sample(5)
train['review'] = train['review'].apply(lambda x: BeautifulSoup(x, 'lxml').get_text().strip())
train.iloc[6005][0]
train.sample(5)
def RemoveSpecialChars(x):
    
    x = re.sub(r'[^\w ]+', "", x)
    x = ' '.join(x.split())
    return x
train['review'] = train['review'].apply(lambda x: RemoveSpecialChars(x))
train.sample(5)
train.iloc[6005][0]
def lemme(x):
    
    x = str(x)
    x_list = []
    doc = nlp(x)
    
    for token in doc:
        lemma = token.lemma_
        
        if lemma == '-PRON-' or lemma == 'be':
            lemma = token.text
            
        x_list.append(lemma)
        
    return ' '.join(x_list)
%%time
train['review'] = train['review'].apply(lambda x: lemme(x))
train.sample(5)
stopwords
len(stopwords)
def RemoveStopWords(x):
    
    return ' '.join([word for word in x.split() if word not in stopwords])
x = train.iloc[6005][0]
# EXAMPLE CODE

print(x)
print()
print("length of x: ",len(x))
x1 = RemoveStopWords(x)
x1
len(x1)
%%time

train['review'] = train['review'].apply(lambda x: RemoveStopWords(x))
train.sample(5)
text = ' '.join(train['review'])
#text
len(text)
# Creating Frequency

text_series = pd.Series(text.split())
freq_comm = text_series.value_counts()
freq_comm
rare_words = freq_comm[-82000:-1]
'rockumentarie' in rare_words
rare_words
# Removing 82000 rare occuring words 

train['review'] = train['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in rare_words]))
train['review'].sample(5)
train['sentiment'].value_counts()
X = train['review']
y = train['sentiment']
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)
X.shape
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4, stratify = y)
X_train.shape, X_test.shape
%%time

#tsvd = TSVD(n_components=10000, random_state=4)
#X_train_tsvd = tsvd.fit_transform(X_train)
#sum(tsvd.explained_variance_)
#clf_svc = SVC()
%%time

#scores = cross_val_score(clf_svc, X_train, y_train, cv=6, n_jobs=-1)
#scores
from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
X_train
%%time

scores = cross_val_score(clf_lr, X_train, y_train, cv=10, n_jobs=4)
scores
scores.mean()
clf_lr.fit(X_train, y_train)
y_test_pred = clf_lr.predict(X_test)
print(classification_report(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)
clf_lr.predict(tfidf.transform(['American Psycho deserved an Oscar, they were robbed']))
y_real_pred = clf_lr.predict(tfidf.transform(test['review']))
print(classification_report(test['sentiment'], y_real_pred))
clf_lr.predict(tfidf.transform(["What hell was that, it's a masterpiece"]))








