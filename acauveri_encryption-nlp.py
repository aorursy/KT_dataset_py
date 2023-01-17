

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

import os

print(os.listdir("../input"))

from sklearn.feature_extraction.text import TfidfVectorizer
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")

samp=pd.read_csv("../input/sample_submission.csv")
test.head(2)
train.head()
train['difficulty'].value_counts()
train.info()
print("Total length of newsgroup :",len(train['target'].value_counts()))
tf=TfidfVectorizer()

train_inp=tf.fit_transform(train['ciphertext'])
train_inp.shape
rf=RandomForestClassifier()
rf.fit(train_inp,train['target'])
test_inp=tf.transform(test['ciphertext'])
pre=pd.DataFrame(rf.predict(test_inp),columns=['Predicted'])
pd.concat([test[['Id']],pre],axis=1).to_csv('submit.csv',index=False)