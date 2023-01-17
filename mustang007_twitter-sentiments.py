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
dataset = pd.read_csv('/kaggle/input/twitter-sentiments-data/train_2kmZucJ.csv')
dataset.head()
test_data = pd.read_csv('/kaggle/input/test-data/test_oJQbWVk.csv', usecols=['tweet'])
test_data.tail()
dataset.info()
dataset.groupby(['label'])['label'].count().plot.bar(color='orange')
dataset['result'] = dataset['label'].astype('O')
data = dataset.copy()
data.drop(columns='label', inplace=True)
data.drop(columns='id', inplace=True)
data.head()
data.describe()
data.drop_duplicates(inplace=True)
data.info()
data['result'] = data['result'].astype('int')
data.info()
import string
import nltk
from nltk.corpus import stopwords
stopword = stopwords.words('english')
def text_preprocessing(texts):
    tex=texts.strip()
    texts_word=[word for word in tex.split() if "@" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "#" not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "www." not in word]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if "http" not in word]
    tex=" ".join(texts_word)
    
    texts_word=[word for word in tex if word not in string.punctuation]
    tex="".join(texts_word)
    texts_word=[word for word in tex.split() if word not in stopword]
    tex=" ".join(texts_word)
    texts_word=[word for word in tex.split() if word.isalpha()]
    tex=" ".join(texts_word)
    texts_word=[word.lower() for word in tex.split()]
    tex=" ".join(texts_word)
    tex=tex.strip()
    return tex.split()
    
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
data['tweet'] = data['tweet'].astype('str')
test_data['tweet'] = test_data['tweet'].astype('str')
cv=CountVectorizer(analyzer=text_preprocessing).fit(data.tweet)
test_cv=CountVectorizer(analyzer=text_preprocessing).fit(test_data.tweet)
cv_trans=cv.transform(data.tweet)
cv_trans
test_cv_trans=cv.transform(test_data.tweet)
test_cv_trans
tfidf=TfidfTransformer().fit(cv_trans)
tfidf_trans=tfidf.transform(cv_trans)
test_tfidf=TfidfTransformer().fit(test_cv_trans)
test_tfidf_trans=test_tfidf.transform(test_cv_trans)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
X_train,X_test,y_train,y_test=train_test_split(tfidf_trans,data.result,test_size=0.15,random_state=42)
# from sklearn.model_selection import GridSearchCV
# C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
# gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# kernel=['rbf','linear']
# hyper={'kernel':kernel,'C':C,'gamma':gamma}
# grid = GridSearchCV(SVC(),param_grid=hyper, verbose=100)

# # gd=GridSearchCV(estimator=SVC(),param_grid=hyper,verbose=True)
# # gd.fit(X_train,y_train)
# # print(gd.best_score_)
# # print(gd.best_estimator_)
# grid.fit(X_train,y_train)   
# print(grid.best_score_)
# print(grid.best_estimator_)
# sv_model =SVC(C=1,gamma=0.2, kernel='linear
              
sv_model = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.7, kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


sv_model.fit(X_train, y_train)
confusion_matrix(y_test,sv_model.predict(X_test))
from sklearn.metrics import f1_score,average_precision_score,roc_auc_score

print(roc_auc_score(y_test, sv_model.predict(X_test)))
print(f1_score(y_test, sv_model.predict(X_test)))
print(average_precision_score(y_test, sv_model.predict(X_test)))
pred = sv_model.predict(X_test)
# for i in pred:
#     print(i)
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

model.fit(X_train,y_train)
from sklearn.metrics import f1_score,average_precision_score,roc_auc_score

print(roc_auc_score(y_test, model.predict(X_test)))
print(f1_score(y_test, model.predict(X_test)))
print(average_precision_score(y_test, model.predict(X_test)))
test_pred = sv_model.predict(test_tfidf_trans)
print(len(test_pred))
# for i in test_pred:
#     print(i)