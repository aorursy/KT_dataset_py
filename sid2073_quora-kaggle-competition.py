# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
(data.shape,test.shape)
data.head()
from wordcloud import WordCloud
import matplotlib.pyplot as plt
doc=data['question_text'].str.lower()
doc_clean=doc.str.replace('[^a-z ]','')
sincere_index=data[data['target']==0].index
wc=WordCloud().generate(' '.join(doc_clean.loc[sincere_index]))
plt.imshow(wc)
doc=data['question_text'].str.lower()
doc_clean=doc.str.replace('[^a-z ]','')
insincere_index=data[data['target']==1].index
wc=WordCloud().generate(' '.join(doc_clean.loc[insincere_index]))
plt.imshow(wc)
from sklearn.model_selection import train_test_split

train,validate=train_test_split(data,test_size=0.3,random_state=100)
(train.shape, validate.shape)
train['target'].value_counts()/train.shape[0],validate['target'].value_counts()/validate.shape[0]
from sklearn.feature_extraction.text import CountVectorizer
dtm_func=CountVectorizer(stop_words='english',min_df=.5)
dtm_func.fit(doc_clean.loc[train.index])
dtm_matrix=dtm_func.transform(doc_clean.loc[train.index])
validate_dtm_matrix=dtm_func.transform(doc_clean.loc[validate.index])
train_x=dtm_matrix
train_y=train['target']
validate_x=validate_dtm_matrix
validate_y=validate['target']
validate_dtm_matrix
from sklearn.ensemble import RandomForestClassifier
dt_model=RandomForestClassifier(n_estimators=300,max_depth=30,random_state=100)
dt_model.fit(train_x,train_y)
pred_validate=dt_model.predict_proba(validate_x)
threshold=[]
f1_score1=[]
recall=[]
for i in np.arange(0.1,0.9,.01):
    threshold.append(i)
    x=np.where(pred_validate>i,1,0)
    score=f1_score(validate_y,x[:,1])
    recall.append(metrics.recall_score(validate_y,x[:,1]))
    f1_score1.append(score)
plt.plot(threshold,f1_score1)
plt.plot(threshold,recall,label='recall')
plt.legend(loc='best')
pred_validate=dt_model.predict_proba(validate_x)
x=np.where(pred_validate>.133034,1,0)
pd.Series(x[:,1]).value_counts()
validate_y.value_counts()
doc_clean_test=test['question_text'].str.lower().str.replace('[^a-z ]','')
test_dtm_matrix=dtm_func.transform(doc_clean_test)
pred_test=dt_model.predict_proba(test_dtm_matrix)
submission=pd.Series(np.where(pred_test[:,1]>.133034,1,0))
submission=pd.DataFrame({'qid':test['qid'],'prediction':submission})
submission.to_csv('submission.csv',index=False)





