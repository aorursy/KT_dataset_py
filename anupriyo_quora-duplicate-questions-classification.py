# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train=pd.read_csv('/kaggle/input/quora-question-pairs/train.csv.zip')

train.head()

test=pd.read_csv('/kaggle/input/quora-question-pairs/test.csv.zip')

test=test[1:20000]

test1=test.copy()

train=train[1:40000]

train['is_duplicate'].value_counts()

#We can see there is an imbalance in predicted class set.

#Lets remove the imbalance by upsampling/downsampling the class variables

#Upsampling

from sklearn.utils import resample

duplicate_ids=train[train['is_duplicate']==1]

duplicate_ids

not_duplicate_ids=train[train['is_duplicate']==0]

duplicate_upsampled=resample(duplicate_ids,replace=True,n_samples=len(not_duplicate_ids),random_state=27)

upsampled=pd.concat([not_duplicate_ids,duplicate_upsampled])

upsampled['is_duplicate'].value_counts()

upsampled['question1']=upsampled['question1'].astype(str)

upsampled['question2']=upsampled['question2'].astype(str)

#Lets clean up the texts

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def clean_text(text):

    text=text.lower()

    text=re.sub('[^a-zA-Z0-9\s]','',text)

    text=text.split()

    text=[ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

    reviews=' '.join(text)

    return reviews

upsampled['question1']=upsampled['question1'].apply(lambda x:clean_text(x))

upsampled['question1']

upsampled['question2']=upsampled['question2'].apply(lambda x:clean_text(x))

test1['question1']=test1['question1'].apply(lambda x:clean_text(x))

test1['question2']=test1['question2'].apply(lambda x:clean_text(x))

#TF-IDF vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(analyzer='word',max_features=300,norm='l1')

tfidf_question1=tfidf.fit_transform(upsampled['question1'])

tfidf_question2=tfidf.fit_transform(upsampled['question2'])

test_tfidf_question1=tfidf.fit_transform(test['question1'])

test_tfidf_question2=tfidf.fit_transform(test['question2'])

#Finding the difference of the vocabulary matrix

train_x=abs(tfidf_question1- tfidf_question2)

train_x.shape

test_x1=abs(test_tfidf_question1-test_tfidf_question2)

test_x1.shape

Y=upsampled['is_duplicate']

Y.shape

from sklearn.model_selection import train_test_split,RandomizedSearchCV

train_x,test_x,train_y,test_y=train_test_split(train_x,Y,test_size=0.2,random_state=10)

train_x.shape

train_y.shape

test_y.shape

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint as sp_randint

rf=RandomForestClassifier(n_estimators=150,n_jobs=-1)

model=rf.fit(train_x,train_y)

pred=model.predict(test_x)

pred

from sklearn.metrics import accuracy_score,f1_score,recall_score

accuracy_score(test_y,pred)

f1_score(test_y,pred)

recall_score(test_y,pred)

#recall_score(test,y_pred)

#recall_score-0.79

#f1_score-0.78

pred1=rf.predict(test_x1)

submission = pd.DataFrame({'Question1':test['question1'],'Question2':test['question2'],'is_duplicate':pred1})

submission.head()

submission[submission['is_duplicate']==1]

type(train_y)

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

params={'objective':'binary:logistic',

        'eval_metric':'auc',

        'eta':0.01,

        'max_depth':100,

        'subsample':0.6,

        'alpha':0.01,

        'random_state':10,

        'device':'gpu'

       }

tr_x=xgb.DMatrix(train_x,train_y)

tr_x

ts_x=xgb.DMatrix(test_x)

ts_x1=xgb.DMatrix(test_x1)

model=xgb.train(params,tr_x,2000)

pred_xg=model.predict(ts_x)

pred_xg

for i in range(0,10044):

    if pred_xg[i]>0.5:

        pred_xg[i]=1

    else:

        pred_xg[i]=0

accuracy_score(test_y,pred_xg)

f1_score(test_y,pred_xg)

recall_score(test_y,pred_xg)

pred_xg1=model.predict(ts_x1)

for i in range(0,19999):

    if pred_xg1[i]>0.5:

        pred_xg1[i]=1

    else:

        pred_xg1[i]=0



submission = pd.DataFrame({'Question1':test['question1'],'Question2':test['question2'],'is_duplicate':pred_xg1})

submission.head()

submission[submission['is_duplicate']==1]

#Recall Score-0.85

#F-Score-0.80

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

os.listdir('../input')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session