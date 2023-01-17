# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.head()
train_df.shape
train_df.isnull().sum()
train_df['target'].value_counts()
wordnet=WordNetLemmatizer()
def clean_text(x):
    text=re.sub('[^A-Za-z]',' ',x)
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[wordnet.lemmatize(word) for word in text]
    text=' '.join(text)
    return text
train_df['text']=train_df['text'].apply(clean_text)
train_df.head()
x=train_df['text']
x
y=train_df['target']
y
count_vect=CountVectorizer(stop_words='english')
tfidf_vect=TfidfVectorizer(stop_words='english')
X1=count_vect.fit_transform(x)
X2=tfidf_vect.fit_transform(x)
cv=ShuffleSplit(n_splits=10,test_size=0.3,random_state=42)
cross_val_score(MultinomialNB(),X1,y,cv=cv,scoring='accuracy')
cross_val_score(MultinomialNB(),X1,y,cv=cv,scoring='accuracy').mean()
cross_val_score(XGBClassifier(),X1,y,cv=cv,scoring='accuracy')
cross_val_score(XGBClassifier(),X1,y,cv=cv,scoring='accuracy').mean()
cross_val_score(SVC(),X1,y,cv=cv,scoring='accuracy')
cross_val_score(SVC(),X1,y,cv=cv,scoring='accuracy').mean()
cross_val_score(MultinomialNB(),X2,y,cv=cv,scoring='accuracy')
cross_val_score(MultinomialNB(),X2,y,cv=cv,scoring='accuracy').mean()
cross_val_score(XGBClassifier(),X2,y,cv=cv,scoring='accuracy')
cross_val_score(XGBClassifier(),X2,y,cv=cv,scoring='accuracy').mean()
cross_val_score(SVC(),X2,y,cv=cv,scoring='accuracy')
cross_val_score(SVC(),X2,y,cv=cv,scoring='accuracy').mean()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)
train_mat=count_vect.fit_transform(x_train)
svc=SVC()
svc.fit(train_mat,y_train)
test_mat=count_vect.transform(x_test)
prediction=svc.predict(test_mat)
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))
tweet_mat=count_vect.fit_transform(x)
model=SVC()
y
tweet_mat.shape
model.fit(tweet_mat,y)
test_df.head()
test_df['text']=test_df['text'].apply(clean_text)
test_df.head()
predict_text=test_df['text']
predict_text
X_text=count_vect.transform(predict_text)
X_text.shape
prediction=model.predict(X_text)
test_df['target']=prediction
test_df.head()
submission_df=test_df[['id','target']]
submission_df
submission_df.to_csv('submission.csv',index = False)