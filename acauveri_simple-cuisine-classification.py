

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import LogisticRegression

import os

print(os.listdir("../input"))

from sklearn.metrics import accuracy_score
test=pd.read_json("../input/test.json")

train=pd.read_json("../input/train.json")
print("Train",train.shape)

print("Test",test.shape)
train.head()
test.head()
pd.merge(train,test,on='id',how='inner').shape[0]
clf=Pipeline([('countV',CountVectorizer()),

          ('tf',TfidfTransformer()),

          ('lg',LogisticRegression())])
train['ingredients_new']=train['ingredients'].apply(lambda x:' '.join(x))
clf.fit(train['ingredients_new'],train['cuisine'])
accuracy_score(clf.predict(train.ingredients_new),train.cuisine)
test.head()
test['ing_new']=test.ingredients.apply(lambda x:' '.join(x))
type(test[['id']])


pre=pd.DataFrame(clf.predict(test['ing_new']),columns=['cuisine'])

pd.concat([test[['id']],pre],axis=1).to_csv('sub.csv',index=False)