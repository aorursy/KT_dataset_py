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
train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
test_df=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(train_df.isnull())
sns.heatmap(test_df.isnull())
train_df=train_df.drop(['id','location','keyword'],axis=1)
test_df=test_df.drop(['id','location','keyword'],axis=1)
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string
def text_cleaning(text):
    '''
    Make text lowercase, remove text in square brackets,remove links,remove special characters
    and remove words containing numbers.
    '''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) # remove special chars
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text
train_df['text']=train_df['text'].apply(text_cleaning)
test_df['text']=test_df['text'].apply(text_cleaning)
train_df
x=train_df['text']
y=train_df['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(min_df=2,ngram_range=(1,2))
x_train_trans=cv.fit_transform(x_train)


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=500)
lr.fit(x_train_trans,y_train)
pred_y=lr.predict(cv.transform(x_test))
from sklearn.metrics import accuracy_score
score_1=accuracy_score(y_test,pred_y)
score_1
from sklearn.neighbors import KNeighborsClassifier
list_1=[]
for i in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_trans,y_train)
    pred_1=knn.predict(cv.transform(x_test))
    score_2=accuracy_score(y_test,pred_1)
    list_1.append(score_2)
plt.plot(range(1,11),list_1)
plt.show()
print(max(list_1))
from sklearn.naive_bayes import MultinomialNB
clf=MultinomialNB(alpha=1)
clf.fit(x_train_trans,y_train)
pred_3=clf.predict(cv.transform(x_test))
score_3=accuracy_score(y_test,pred_3)
score_3
from sklearn.model_selection import GridSearchCV
params={'C':[100,10,1,0.1,0.001,0.0001],
       'solver' :['newton-cg', 'lbfgs', 'liblinear']}
search_1=GridSearchCV(lr,params,cv=5,verbose=0)
search_1.fit(x_train_trans,y_train)
print(search_1.best_params_)
lr=LogisticRegression(C=0.1,solver='liblinear')
lr.fit(x_train_trans,y_train)
pred_4=lr.predict(cv.transform(x_test))
score_4=accuracy_score(y_test,pred_4)
score_4
from sklearn.svm import SVC


print(search_2.best_params_)
svm=SVC(C=10,gamma=0.01,kernel='rbf')
svm.fit(x_train_trans,y_train)
pred_5=svm.predict(cv.transform(x_test))
score_5=accuracy_score(y_test,pred_5)
score_5

preds=svm.predict(cv.transform(test_df['text']))
submission['target']=preds
submission
submission.to_csv('submission.csv', index=False)