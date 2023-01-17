import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt
adult = pd.read_csv('../input/adult-pmr3508/train_data.csv', names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adultTest = pd.read_csv('../input/adult-pmr3508/test_data.csv', names=["Id",

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',    na_values="?")
adult.head()
adultTest.head()
nadult = adult.dropna()
Xadult = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Xadult.head()
Yadult = nadult.Target
XtestAdult = adultTest[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

XtestAdult.head()
for a in XtestAdult.columns:

    XtestAdult[a].fillna(XtestAdult[a].mode()[0],inplace=True)
Xadult.drop(['Id'])
Xadult.to_csv('Xadult', header=False, index=False)

Xadult = pd.read_csv('Xadult', header=None)
Xadult = Xadult.drop([0])
Yadult = Yadult.drop(['Id'])
XtestAdult = XtestAdult.drop([0])

XtestAdult.head()
XtestAdult.to_csv('XtestAdult', header=False, index=False)

XtestAdult = pd.read_csv('XtestAdult', header=None)
YtestAdult = adultTest.Target
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
best = 0

media_anterior = 0

for i in range(120,130):

    rf = RandomForestClassifier(n_estimators=i)  

    scores = cross_val_score(rf, Xadult, Yadult, cv=10)

    media = sum(scores)/len(scores)

    if media > media_anterior:

        media_anterior = media

        best = i

    print(i)

print(media_anterior)

print(best)
rf = RandomForestClassifier(n_estimators=i) 
rf.fit(Xadult,Yadult)
YtestPredrf = rf.predict(XtestAdult)
Index = []

for j in range( len(YtestPredrf)):

    Index.append(j)
Id = pd.DataFrame(Index)
Predrf = pd.DataFrame(YtestPredrf)

Predrf.columns = ['Income']

Predrf.insert(0, 'Id', Id, True)

Predrf.head()
Predrf.to_csv('AdultPredictionrf.csv')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
log = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=400, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
scoreslog = cross_val_score(log, Xadult, Yadult, cv=10)

medialog = sum(scoreslog)/len(scoreslog)

print(medialog)
log.fit(Xadult,Yadult)
YtestPredlog = log.predict(XtestAdult)
Index = []

for j in range( len(YtestPredlog)):

    Index.append(j)
Id = pd.DataFrame(Index)
Predlog = pd.DataFrame(YtestPredlog)

Predlog.columns = ['Income']

Predlog.insert(0, 'Id', Id, True)

Predlog.head()
Predlog.to_csv('AdultPredictionlog.csv')
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
svc=SVC(gamma='auto')
scoressvc = cross_val_score(svc, Xadult, Yadult, cv=10)

mediasvc = sum(scoressvc)/len(scoressvc)

print(mediasvc)
svc.fit(Xadult,Yadult)
YtestPredsvc = svc.predict(XtestAdult)
Index = []

for j in range( len(YtestPredsvc)):

    Index.append(j)
Id = pd.DataFrame(Index)
Predsvc = pd.DataFrame(YtestPredsvc)

Predsvc.columns = ['Income']

Predsvc.insert(0, 'Id', Id, True)

Predsvc.head()
Predsvc.to_csv('AdultPredictionsvc.csv')
calitrain = pd.read_csv('../input/california/calitrain.csv')
calitest = pd.read_csv('../input/california/calitest.csv')
calitrain.head()
calitest.head()
ncalitrain = calitrain.dropna()
calitrain.shape
ncalitrain.shape
for df in [calitrain,calitest]:

    df.set_index('Id',inplace=True)
Xcalitrain = calitrain.drop(columns='median_house_value')

Ycalitrain = calitrain.median_house_value
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
lin = LinearRegression()
CVlin = cross_val_score(lin, Xcalitrain, Ycalitrain, cv=10)

medialin = sum(CVlin)/len(CVlin)

print(medialin)
lin.fit(Xcalitrain,Ycalitrain)
Ytestlin = lin.predict(calitest)
Ytestlin
Index = []

for j in range( len(Ytestlin)):

    Index.append(j)
Id = pd.DataFrame(Index)
Predlin = pd.DataFrame(Ytestlin)

Predlin.columns = ['median_house_value']

Predlin.insert(0, 'Id', Id, True)

Predlin.head()
Predlin.to_csv('CaliPredictionlin.csv')
from sklearn.linear_model import Lasso

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
lasso = Lasso(alpha = 0.1)
CVlasso = cross_val_score(lasso, Xcalitrain, Ycalitrain, cv=10)

medialasso = sum(CVlasso)/len(CVlasso)

print(medialasso)
lasso.fit(Xcalitrain,Ycalitrain)
Ytestlasso = lasso.predict(calitest)
Index = []

for j in range( len(Ytestlasso)):

    Index.append(j)
Id = pd.DataFrame(Index)
Predlasso = pd.DataFrame(Ytestlasso)

Predlasso.columns = ['median_house_value']

Predlasso.insert(0, 'Id', Id, True)

Predlasso.head()
Predlasso.to_csv('CaliPredictionlasso.csv')
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
rid = Ridge()
CVrid = cross_val_score(rid, Xcalitrain, Ycalitrain, cv=10)

mediarid = sum(CVrid)/len(CVrid)

print(mediarid)
rid.fit(Xcalitrain,Ycalitrain)
Ytestrid = rid.predict(calitest)
Index = []

for j in range( len(Ytestrid)):

    Index.append(j)
Id = pd.DataFrame(Index)
Predrid = pd.DataFrame(Ytestrid)

Predrid.columns = ['median_house_value']

Predrid.insert(0, 'Id', Id, True)

Predrid.head()
Predrid.to_csv('CaliPredictionlasso.csv')