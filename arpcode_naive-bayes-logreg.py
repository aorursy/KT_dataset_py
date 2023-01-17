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
data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
len(data[data['target']==1]); len(data)
data=data.to_numpy()
x_trn,y_trn = data[:,1:-1][:4500],data[:,-1][:4500]

x_vld,y_vld = data[:,1:-1][4500:],data[:,-1][4500:]
len(x_trn), len(x_vld)
x_trn[0]
from sklearn.feature_extraction.text import CountVectorizer

veczr = CountVectorizer(ngram_range=(1,3))
trn_term = veczr.fit_transform(x_trn[:,2])

vld_term = veczr.transform(x_vld[:,2])
vocab = veczr.get_feature_names(); vocab[:100], len(vocab)
trn_term.shape
y_trn=y_trn.astype('int64')

y_vld=y_vld.astype('int64')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score as f1

m=LogisticRegression(C=1e8,dual=True,solver='liblinear',max_iter=400, random_state=1)

m.fit(trn_term,y_trn)

preds = m.predict(vld_term)

(preds==y_vld).mean(), f1(y_vld,preds)
m=LogisticRegression(C=10,dual=True,solver='liblinear',max_iter=400, random_state=1)

m.fit(trn_term,y_trn)

preds = m.predict(vld_term)

(preds==y_vld).mean(), f1(y_vld,preds)
p = trn_term[y_trn==1].sum(0)+1

q = trn_term[y_trn==0].sum(0)+1



r = np.log(((p/p.sum())/(q/q.sum())))
m=LogisticRegression(C=0.1,dual=True,solver='liblinear',max_iter=300, random_state=1)

m.fit(trn_term.multiply(r),y_trn)

preds = m.predict(vld_term.multiply(r))

(preds==y_vld).mean(), f1(y_vld,preds)
x = veczr.fit_transform(data[:,3])

y = data[:,-1]

y
p = x[y==1].sum(0)+1

q = x[y==0].sum(0)+1

p,q
r = np.log((p/p.sum())/(q/q.sum()))

test = veczr.transform(test_data["text"])

test_data["text"]
m=LogisticRegression(C=0.1,dual=True,solver='liblinear',max_iter=500, random_state=69)

m.fit(x.multiply(r),y.astype('int64'))

submission["target"] = m.predict(test.multiply(r))

submission.to_csv("submission1.csv", index=False)
m=LogisticRegression(C=1e8,dual=True,solver='liblinear',max_iter=600, random_state=69)

m.fit(x,y.astype('int64'))

submission["target"] = m.predict(test)

submission.to_csv("submission2.csv", index=False)
# kaggle competitions submit -c nlp-getting-started -f submission.csv -m "Message"