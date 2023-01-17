import pandas as pd

import numpy as np
DF = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')

DF.head()
DF.info()
DF.describe()
np.unique(DF['target_class'])
Features = DF.columns[:-1]

Label = DF.columns[-1]
X = DF[Features].values

y = DF[Label].values
X.shape
y.shape
y.shape
from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test = tts(X,y,stratify=y,random_state =1,test_size=0.2)
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.base import clone

def My_pipe(estimator,Compression = True):

    if Compression:

        return make_pipeline(StandardScaler(),

                        PCA(n_jobs = 2),

                            clone(estimator))

    return make_pipeline(StandardScaler(),

                            clone(estimator))
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

Lr = LogisticRegression(random_state=1,max_iter=10000,solver='liblinear',C=10)

Lr_pipe = My_pipe(estimator=Lr,Compression=False)
scores = cross_val_score(estimator=Lr_pipe,X=X_train,y=y_train,cv=10,scoring='f1')
print(scores.mean(),' +/- ',scores.std())
from sklearn.svm import SVC

sv  =SVC(random_state=1,kernel='linear')

svc_pipe = My_pipe(estimator=sv,Compression=False)
scores = cross_val_score(estimator=svc_pipe,X=X_train,y=y_train,cv=10,scoring='f1')
print(scores.mean(),scores.std())
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=1,n_estimators=100)

forest_pipe = My_pipe(estimator=rfc,Compression=False)
scores_forest = cross_val_score(estimator=forest_pipe,X=X_train,y=y_train,cv=10,scoring='f1',n_jobs=-1)
print(scores_forest.mean(),' +/- ',scores_forest.std())
forest_pipe.fit(X_train,y_train)
y_train_pred = forest_pipe.predict(X_train)

y_pred_test = forest_pipe.predict(X_test)
from sklearn.metrics import f1_score

f1_train_1 = f1_score(y_pred=y_train_pred,y_true=y_train,pos_label=1)

f1_test_1 = f1_score(y_pred=y_pred_test,y_true=y_test,pos_label=1)
print('train : ' ,f1_train_1)

print('test : ',f1_test_1)
f1_train_0 = f1_score(y_pred=y_train_pred,y_true=y_train,pos_label=0)

f1_test_0 = f1_score(y_pred=y_pred_test,y_true=y_test,pos_label=0)
print('train : ' ,f1_train_0)

print('test : ',f1_test_0)
N_train_1 = X_train[y_train==1].shape[0]

n_test_1 = X_test[y_test==1].shape[0]

f1_avg_train = (N_train_1*(f1_train_1) +(X_train.shape[0]-N_train_1)*f1_train_0)/X_train.shape[0]

f1_avg_test = (n_test_1*(f1_test_1) +(X_test.shape[0]-n_test_1)*f1_test_0)/X_test.shape[0]
print('f1 avg train :',(f1_avg_train))

print('f1 avg test :',(f1_avg_test))