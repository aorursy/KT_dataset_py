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
import pandas as pd

import numpy as np

import re

import string

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from xgboost import XGBClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import cross_val_score #Additional scklearn functions

from sklearn.model_selection import GridSearchCV   #Perforing grid search
from keras.preprocessing.text import Tokenizer

from keras.utils.vis_utils import plot_model

from keras.models import Sequential

from keras.layers import Dense
from collections import Counter
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

vocab = Counter()
train  = pd.read_csv('../input/nlp-getting-started/train.csv')

test  = pd.read_csv('../input/nlp-getting-started/test.csv')
#train[train['target']==1]['keyword'].value_counts()
train = train.drop(['keyword','location'],axis=1)

test = test.drop(['keyword','location'],axis=1)
test
train['text']=train['text'].map(lambda x : x.split())

test['text']=test['text'].map(lambda x : x.split())


train['text'] = train['text'].map(lambda x:  [word for word in x if not word.startswith(('http','@'))])

re_punc = re.compile('[%s]'%re.escape(string.punctuation))

train['text'] = train['text'].map(lambda x:  [re_punc.sub('',w) for w in x])

train['text'] = train['text'].map(lambda x: [word for word in x if word.isalpha()])

train['text'] = train['text'].map(lambda x: [w.lower() for w in x])    

train['text'] = train['text'].map(lambda x:  [w for w in x if not w in stop_words ]) 

train['text'] = train['text'].map(lambda x: [w for w in x if len(w)>1])



test['text'] = test['text'].map(lambda x:  [word for word in x if not word.startswith(('http','@'))])

re_punc = re.compile('[%s]'%re.escape(string.punctuation))

test['text'] = test['text'].map(lambda x:  [re_punc.sub('',w) for w in x])

test['text'] = test['text'].map(lambda x: [word for word in x if word.isalpha()])

test['text'] = test['text'].map(lambda x: [w.lower() for w in x])    

test['text'] = test['text'].map(lambda x:  [w for w in x if not w in stop_words ]) 

test['text'] = test['text'].map(lambda x: [w for w in x if len(w)>1])

for line in train['text']:

    vocab.update(line)
#new_voc = vocab.copy()

#for key,val in vocab.items():

 #   if val<5:

  #      del new_voc[key]
o = list(vocab)
line = list(train['text'][0])

line
count =0

arr = []

for line in train['text']:

    vec = np.zeros(len(vocab))

    for word in line:

        

        if vocab[word]!=0:

            vec[o.index(word)]=(line.count(word)/len(line))*np.log(len(train)/vocab[word])

    arr.append(vec)

    count+=1

        #df.append(list(vec))

X_train = np.array(arr)
count =0

arr = []

for line in test['text']:

    vec = np.zeros(len(vocab))

    for word in line:

        

        if vocab[word]!=0:

            vec[o.index(word)]=(line.count(word)/len(line))*np.log(len(test)/vocab[word])

    arr.append(vec)

    count+=1

        #df.append(list(vec))

X_test = np.array(arr)
X_train
X_train.shape
X_train[0:5,0:5]
y_train = list(train['target'])

y_train[0:5]
X_test.shape
train_new=pd.DataFrame(X_train)

train_new['target']=y_train

print(train_new.head())
xgb_classifir = XGBClassifier(learning_rate=0.05,

                              num_round=1000,

                              max_depth=10,

                              min_child_weight=2,

                              colsample_bytree=0.7,

                              subsample=0.8,

                              gamma=0.3,

                              reg_alpha=1e-5,

                              reg_lambda=1,

                              n_estimators=1000,

                              objective='binary:logistic',

                              eval_metric=["auc", "logloss", "error"],

                              early_stopping_rounds=50)
trx, valx, trY, valY = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
xgb_classifir.fit(trx, trY, eval_set=[(trx, trY),(valx, valY)])
y_pred_xgb = xgb_classifir.predict(X_test)
predictions = [round(value) for value in y_pred_xgb]
df = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

df['target']=predictions

df.head()
df.to_csv('submission_xgb.csv',index=False)