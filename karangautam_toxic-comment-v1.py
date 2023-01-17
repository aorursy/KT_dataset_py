# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack
from nltk.corpus import stopwords
stop = stopwords.words('english')
input = pd.read_csv('../input/train.csv')
test= pd.read_csv('../input/test.csv')
input.shape
input.head()
#from sklearn.model_selection import train_test_split
#trainX,testX,trainY,testY = train_test_split(input.drop(['target'],axis=1), input['target'], test_size=0.33, random_state=42)
trainX= input.drop(['target'],axis=1)
trainY= input['target']
testX= test
def tokenizer(text):
    if text:
        result = re.findall('[a-z]{2,}', text.lower())
    else:
        result = []
    return result
testX.head()
trainX['word_count'] = trainX['question_text'].apply(lambda x: len(str(x).split(" ")))
testX['word_count'] = testX['question_text'].apply(lambda x: len(str(x).split(" ")))
trainX['char_count'] = trainX['question_text'].str.len() ## this also includes spaces
testX['char_count'] = testX['question_text'].str.len() 
trainX[['question_text','char_count']].head()
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

trainX['avg_word'] = trainX['question_text'].apply(lambda x: avg_word(x))
testX['avg_word'] = testX['question_text'].apply(lambda x: avg_word(x))
trainX[['question_text','avg_word']].head()
trainX['stopwords'] = trainX['question_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
testX['stopwords'] = testX['question_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
trainX[['question_text','stopwords']].head()
trainX['numerics'] = trainX['question_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
testX['numerics'] = testX['question_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
trainX[['question_text','numerics']].head()
trainX['upper'] = trainX['question_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
testX['upper'] = testX['question_text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
trainX[['question_text','upper']].head()
trainX['qmark']=trainX[['question_text']].applymap(lambda x: str.count(x, '?'))
trainX['esclampoint']=trainX[['question_text']].applymap(lambda x: str.count(x, '!'))
trainX['atrateof']=trainX[['question_text']].applymap(lambda x: str.count(x, '@'))
testX['qmark']=testX[['question_text']].applymap(lambda x: str.count(x, '?'))
testX['esclampoint']=testX[['question_text']].applymap(lambda x: str.count(x, '!'))
testX['atrateof']=testX[['question_text']].applymap(lambda x: str.count(x, '@'))
trainX1 = trainX['question_text']
testX1 = testX['question_text']
import time
import re
vect = TfidfVectorizer(tokenizer=tokenizer, stop_words='english',max_features=5000)
start = time.time()
X_train_vect = vect.fit_transform(trainX1)
X_test_vect = vect.transform(testX1)
end = time.time()
print('Time to train vectorizer and transform training text: %0.2fs' % (end - start))
X_train_vect
trainX.head().T
import scipy
sparse_matrix= scipy.sparse.csr_matrix(trainX.drop(['qid','question_text'],axis=1))
sparse_matrix1= scipy.sparse.csr_matrix(testX.drop(['qid','question_text'],axis=1))

new_features = scipy.sparse.hstack((X_train_vect,sparse_matrix))
test_new_features = scipy.sparse.hstack((X_test_vect,sparse_matrix1))
import xgboost as xgb
#model = SGDRegressor(loss='squared_loss', penalty='l2', random_state=seed, max_iter=5)
#sgd = SGDClassifier(loss="hinge", penalty="l2")
start = time.time()
gbm = xgb.XGBClassifier(max_depth=200, n_estimators=100, learning_rate=0.1,silent=False,n_jobs=-1).fit(new_features, trainY)#,
                                                                                    #       eval_set=[(test_new_features, testY)],
                                                                                    #       eval_metric = ['logloss', 'auc'],early_stopping_rounds=25,
                                                                                    #       verbose=True)
end = time.time()
print('Time to train model: %0.2fs' % (end -start))
model = gbm.best_iteration
y_pred = gbm.predict(test_new_features)
from sklearn.metrics import f1_score

#f1_score(y_pred,testY)
test['prediction']= pd.DataFrame(y_pred)
test.head()
test[['qid','prediction']].to_csv('submission.csv',index=False)
