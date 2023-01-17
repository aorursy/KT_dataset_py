from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score,classification_report

import seaborn as sns

import pandas as pd

import numpy as np 

submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")



labels = train['target'].values

train_id = train.pop("id")

test_id = test.pop("id")
data = pd.concat([train.drop('target',axis=1), test])

totaal=train.append(test)
train.shape,test.shape,data.shape,totaal.shape
data.head()
# One Hot Encode target mean()

cols=[ci for ci in train.columns if ci not in ['id','index','target']]

coltype=train.dtypes

for ci in cols:

    

    if (coltype[ci]=="object"):

        #bin_3

        #l_enc = LabelEncoder()

        codes=totaal[[ci,'target']].groupby(ci).mean().sort_values("target")

        #print(codes)

        codesdict=codes.target.to_dict()



        #print(codesdict)

        #l_enc.fit(list(codes.index))

        totaal[ci]=totaal[ci].map(codesdict) #l_enc.transform(totaal[ci])

    #print('labelized',ci)

#prevent error in test, because nom_8 can have empties

totaal['id']=train_id.append(test_id)

totaal=totaal.fillna(0)
columns = [i for i in test.columns]



dummies = pd.get_dummies(data,

                         columns=columns,

                         drop_first=True,

                         sparse=True,dtype='float')



#del data
from scipy.sparse import coo_matrix, hstack

from scipy.sparse import lil_matrix

sparse_matrix = lil_matrix((len(totaal), len(totaal.columns)))

for k, column_name in enumerate(data.columns):

    sparse_matrix[totaal.id.values, np.full(len(totaal), k)] = totaal[column_name].values

sparse_matrix=sparse_matrix.tocsr()
dummies=dummies.sparse.to_coo().tocsr()
from scipy.sparse import hstack



totdum=hstack(  ( sparse_matrix,dummies ) )

#totdum=pd.concat([totaal.to_sparse(),dummies],ignore_index=True)

totdum.shape,totaal.shape,dummies.shape
train = totdum.tocsc()[:train.shape[0], :]

test = totdum.tocsc()[train.shape[0]:, :]



del dummies,totdum,sparse_matrix,totaal,data
print(train.shape)

print(test.shape)
#train = train.sparse.to_coo().tocsr()

#test = test.sparse.to_coo().tocsr()



train = train.astype("float32")

test = test.astype("float32")
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier,RidgeClassifierCV,LogisticRegression

lr = LogisticRegression( solver="lbfgs",max_iter=500,n_jobs=4)

lengte=100000

lr.fit(train[:lengte], labels[:lengte])



lr_pred = lr.predict_proba(train[:lengte])[:, 1]

score = roc_auc_score(labels[:lengte], lr_pred)

print("score: ", score)



lr_pred = lr.predict(train[:lengte])

score = classification_report(labels[:lengte], lr_pred)

print("score: ", score)



import random



print("First Random float number: ", random.random())

lr_pred = [ (random.random()>0.7)*1 for x in range(lengte) ]

score = classification_report(labels[:lengte], lr_pred)

print("score: ", score)

submission["id"] = test_id

submission["target"] = lr.predict_proba(test)[:, 1]
submission.head()
submission.to_csv("submission.csv", index=False)