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
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

f1=pd.read_json('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset.json',lines=True)
f2=pd.read_json('/kaggle/input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json',lines=True)
df=pd.concat([f1,f2],axis=0)
df.head()
df.info()

print('0 : ',df[df['is_sarcastic']==0].count())
print('1 : ',df[df['is_sarcastic']==1].count())
from sklearn.ensemble import RandomForestClassifier
X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
X_train,X_test,y_train,y_test=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',RandomForestClassifier())])
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.linear_model import LogisticRegression
X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)
X_train,X_test,y_train,y_test=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',LogisticRegression())])
model=pipe.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
from sklearn.svm import SVC,LinearSVC
X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)
model=LinearSVC()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',LinearSVC())])
pipe.fit(X_train_,y_train_)
y_pred=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred))
grid_param={'C':[0.1,1,10],
           'gamma':[1,0.1,0.001]}
grid=GridSearchCV(SVC(),param_grid=grid_param,refit=True,verbose=3)
grid.fit(X_train,y_train)
model=SVC(C=1,gamma=1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.linear_model import SGDClassifier
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',SGDClassifier())])
model=pipe.fit(X_train_,y_train_)
y_pred_=model.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))
from sklearn.naive_bayes import MultinomialNB
X=TfidfVectorizer().fit_transform(df['headline'])
X_train,X_test,y_train,y_test=train_test_split(X,df['is_sarcastic'],test_size=0.2,random_state=101)
model=MultinomialNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],df['is_sarcastic'],test_size=0.2,random_state=101)
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',MultinomialNB())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))
from sklearn.naive_bayes import BernoulliNB
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',BernoulliNB())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))
para={'alpha':[1,0.5,0.1,0.01,0]}
model=GridSearchCV(estimator=BernoulliNB(),param_grid=para,n_jobs=-1,cv=3,verbose=3)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.ensemble import GradientBoostingClassifier
X_train,X_test,y_train,y_test=train_test_split(TfidfVectorizer().fit_transform(df['headline']),
                                              df['is_sarcastic'],
                                              test_size=0.2,
                                               random_state=101)
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],
                                                   df['is_sarcastic'],
                                                   test_size=0.2,
                                                   random_state=101)
model=GradientBoostingClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
pipe=Pipeline([('vect',CountVectorizer()),
             ('tdidf',TfidfTransformer()),
             ('model',GradientBoostingClassifier())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))
from xgboost import XGBClassifier
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',XGBClassifier())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))
from sklearn.tree import DecisionTreeClassifier
X_train,X_test,y_train,y_test=train_test_split(TfidfVectorizer().fit_transform(df['headline']),
                                              df['is_sarcastic'],
                                              test_size=0.2,
                                              random_state=101)
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],df['is_sarcastic'],
                                                  test_size=0.2,
                                                  random_state=101)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
params = {'max_leaf_nodes': list(range(2, 100,10)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)
grid_search_cv.best_params_
model=grid_search_cv.best_estimator_
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',DecisionTreeClassifier())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))
from sklearn.neighbors import KNeighborsClassifier
X_train,X_test,y_train,y_test=train_test_split(TfidfVectorizer().fit_transform(df['headline']),
                                              df['is_sarcastic'],
                                              test_size=0.2,
                                              random_state=101)
X_train_,X_test_,y_train_,y_test_=train_test_split(df['headline'],
                                                  df['is_sarcastic'],
                                                  test_size=0.2,random_state=101)
model=KNeighborsClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
grid=GridSearchCV(estimator=KNeighborsClassifier(),param_grid={'n_neighbors':[1,2,10,20,50]},verbose=3,n_jobs=1)
grid.fit(X_train,y_train)
model=grid.best_estimator_
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print(accuracy_score(y_test,y_pred))
pipe=Pipeline([('vect',CountVectorizer()),
              ('tfidf',TfidfTransformer()),
              ('model',KNeighborsClassifier())])
pipe.fit(X_train_,y_train_)
y_pred_=pipe.predict(X_test_)
print(accuracy_score(y_test_,y_pred_))
