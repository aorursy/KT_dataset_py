# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from xgboost import XGBRegressor as XGBR

from sklearn.ensemble import RandomForestRegressor as RFR

from sklearn.linear_model import LinearRegression as LR

from sklearn.datasets import load_boston

from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS

from sklearn.metrics import mean_squared_error as MSE

import matplotlib.pyplot as plt
data=load_boston()
x=data.data

y=data.target
x.shape
xtrain,xtest,ytrain,ytest=TTS(x,y,test_size=0.3,random_state=0)
reg=XGBR(n_estimators=100,objective ='reg:squarederror').fit(xtrain,ytrain)

reg.score(xtest,ytest)
reg.feature_importances_
reg=XGBR(n_estimators=100,objective ='reg:squarederror')

print(CVS(reg,xtrain,ytrain,cv=5).mean())

print(reg.fit(xtrain,ytrain).score(xtest,ytest))
rfr=RFR(n_estimators=100)

CVS(rfr,xtrain,ytrain,cv=5).mean()
cv = KFold(n_splits=5, shuffle = True, random_state=0)
axisx = range(10,300,10)

rs = []

var = []

ge = []

for i in axisx:

    reg = XGBR(n_estimators=i,random_state=0,objective ='reg:squarederror') 

    

    cvresult = CVS(reg,xtrain,ytrain,cv=cv) 

    

    

    rs.append(cvresult.mean())



    var.append(cvresult.var())

    #genelization error

    ge.append((1 - cvresult.mean())**2+cvresult.var())

    

print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))]) 



print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var)) 



print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge)) 

plt.figure(figsize=(20,5))

plt.plot(axisx,rs,c="black",label="XGB")

rs=np.array(rs)

var=np.array(var)

plt.plot(axisx,rs-var,linestyle='-.',c="red")

plt.plot(axisx,rs+var,linestyle='-.',c='red')

plt.legend()

plt.show() 
axisx = np.linspace(0.8,0.9,20)

rs = []

for i in axisx:

    reg = XGBR(n_estimators=290,subsample=i,random_state=0,objective ='reg:squarederror')

    rs.append(CVS(reg,xtrain,ytrain,cv=cv).mean())

print(axisx[rs.index(max(rs))],max(rs))

plt.figure(figsize=(20,5))

plt.plot(axisx,rs,c="red",label="XGB")

plt.legend()

plt.show()
reg = XGBR(n_estimators=290

           ,subsample=0.868421052631579

           ,random_state=0

          ,objective ='reg:squarederror').fit(xtrain,ytrain)

print(CVS(reg,xtrain,ytrain,cv=cv).mean())

print(reg.fit(xtrain,ytrain).score(xtest,ytest))

print(MSE(ytest,reg.predict(xtest)))
axisx = np.linspace(0.1,0.15,20)

rs = []

for i in axisx:

    reg = XGBR(n_estimators=290,subsample=0.868421052631579,learning_rate=i,random_state=0,objective ='reg:squarederror')

    rs.append(CVS(reg,xtrain,ytrain,cv=cv).mean())

print(axisx[rs.index(max(rs))],max(rs))

plt.figure(figsize=(20,5))

plt.plot(axisx,rs,c="red",label="XGB")

plt.legend()

plt.show()
for booster in ["gbtree","gblinear","dart"]:

    reg = XGBR(n_estimators=290

               ,subsample=0.868421052631579

               ,learning_rate=0.13157894736842105

               ,booster=booster

               ,random_state=0

               ,objective ='reg:squarederror')

    print(booster)

    print(CVS(reg,xtrain,ytrain,cv=cv).mean())
reg = XGBR(n_estimators=290

               ,subsample=0.868421052631579

               ,learning_rate=0.13157894736842105

               ,booster="gbtree"

               ,random_state=0

               ,objective ='reg:squarederror')
import xgboost as xgb

dtrain = xgb.DMatrix(xtrain,ytrain)

dtest = xgb.DMatrix(xtest,ytest)
param = {'silent':True,'objective':'reg:squarederror'

         ,"eta":0.13157894736842105

         ,'subsample':0.868421052631579

         ,'random_state':0

         ,'xgb_model':'gbtree'} 

num_round = 290  

bst = xgb.train(param, dtrain, num_round)

print(CVS(reg,xtrain,ytrain,cv=cv).mean())
axisx = np.arange(0,5,0.05)

rs = []

var = []

ge = []

for i in axisx:

    reg = XGBR(n_estimators=290

               ,subsample=0.868421052631579

               ,learning_rate=0.13157894736842105

               ,booster="gbtree"

               ,random_state=0

               ,objective ='reg:squarederror'

              ,gamma=i)

    result = CVS(reg,xtrain,ytrain,cv=cv)

    rs.append(result.mean())

    var.append(result.var())

    ge.append((1 - result.mean())**2+result.var())

print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))

rs = np.array(rs)

var = np.array(var)*0.1

plt.figure(figsize=(20,5))

plt.plot(axisx,rs,c="black",label="XGB")

plt.plot(axisx,rs+var,c="red",linestyle='-.')

plt.plot(axisx,rs-var,c="red",linestyle='-.')

plt.legend()

plt.show()
import xgboost as xgb 

dfull = xgb.DMatrix(x,y)



param1 = {'silent':True,'obj':'reg:squarederror',"gamma":0,"eval_metric":"rmse"} 

num_round = 290

n_fold=5



cvresult1 = xgb.cv(param1, dfull, num_round,n_fold)

plt.figure(figsize=(20,5))

plt.grid()

plt.plot(range(1,291),cvresult1.iloc[:,0],c="red",label="train,gamma=0")

plt.plot(range(1,291),cvresult1.iloc[:,2],c="orange",label="test,gamma=0")

plt.legend()

plt.show()
cvresult1
param1 = {'silent':True,'obj':'reg:linear',"gamma":0}

param2 = {'silent':True,'obj':'reg:linear',"gamma":20}# Decrese the performance in the train data

num_round = 290

n_fold=5



cvresult1 = xgb.cv(param1, dfull, num_round,n_fold)





cvresult2 = xgb.cv(param2, dfull, num_round,n_fold)



plt.figure(figsize=(20,5))

plt.grid()

plt.plot(range(1,291),cvresult1.iloc[:,0],c="red",label="train,gamma=0")

plt.plot(range(1,291),cvresult1.iloc[:,2],c="orange",label="test,gamma=0")

plt.plot(range(1,291),cvresult2.iloc[:,0],c="green",label="train,gamma=20")

plt.plot(range(1,291),cvresult2.iloc[:,2],c="blue",label="test,gamma=20")

plt.legend()

plt.show()
dfull = xgb.DMatrix(x,y)

param1 = {'silent':True 

          ,'obj':'reg:linear' 

          ,"subsample":1

          ,"max_depth":6

          ,"eta":0.3

          ,"gamma":0

          ,"lambda":1

          ,"alpha":0

          ,"colsample_bytree":1

          ,"colsample_bylevel":1

          ,"colsample_bynode":1

          ,"nfold":5}

num_round = 200
cvresult1 = xgb.cv(param1, dfull, num_round)

fig,ax = plt.subplots(1,figsize=(15,10))

ax.set_ylim(top=5)

ax.grid()

ax.plot(range(1,201),cvresult1.iloc[:,0],c="red",label="train,original")

ax.plot(range(1,201),cvresult1.iloc[:,2],c="orange",label="test,original")

ax.legend(fontsize="xx-large")

plt.show()
dfull = xgb.DMatrix(xtrain,ytrain)

#original

param1 = {'silent':True 

          ,'obj':'reg:linear' 

          ,"subsample":1

          ,"max_depth":6

          ,"eta":0.3

          ,"gamma":0

          ,"lambda":1

          ,"alpha":0

          ,"colsample_bytree":1

          ,"colsample_bylevel":1

          ,"colsample_bynode":1

          ,"nfold":5}

num_round = 200

cvresult1 = xgb.cv(param1, dfull, num_round)

fig,ax = plt.subplots(1,figsize=(15,10))

ax.set_ylim(top=6)

ax.grid()

ax.plot(range(1,201),cvresult1.iloc[:,0],c="red",label="train,original")

ax.plot(range(1,201),cvresult1.iloc[:,2],c="orange",label="test,original")



#adjust 1

param2 = {'silent':True

           ,'obj':'reg:linear' 

          ,"subsample":1

          ,"max_depth":4

          ,"eta":0.2

          ,"gamma":0

          ,"lambda":1

          ,"alpha":0

          ,"colsample_bytree":1

          ,"colsample_bylevel":1

          ,"colsample_bynode":1

          ,"nfold":5}

#adjust 2

param3 = {'silent':True

           ,'obj':'reg:linear' 

          ,"subsample":0.8

          ,"max_depth":4

          ,"eta":0.2

          ,"gamma":0

          ,"lambda":1

          ,"alpha":0

          ,"colsample_bytree":1

          ,"colsample_bylevel":1

          ,"colsample_bynode":1

          ,"nfold":5}



cvresult2 = xgb.cv(param2, dfull, num_round)

cvresult3 = xgb.cv(param3, dfull, num_round)

ax.plot(range(1,201),cvresult2.iloc[:,0],c="green",label="train,last")

ax.plot(range(1,201),cvresult2.iloc[:,2],c="blue",label="test,last")

ax.plot(range(1,201),cvresult3.iloc[:,0],c="gray",label="train,this")

ax.plot(range(1,201),cvresult3.iloc[:,2],c="pink",label="test,this")

ax.legend(fontsize="xx-large")

plt.show()
from sklearn.metrics import mean_squared_error

from time import time

time0=time()

param3 = {'silent':True

           ,'obj':'reg:linear' 

          ,"subsample":0.8

          ,"max_depth":4

          ,"eta":0.2

          ,"gamma":0

          ,"lambda":1

          ,"alpha":0

          ,"colsample_bytree":1

          ,"colsample_bylevel":1

          ,"colsample_bynode":1

          ,"nfold":5}



d_train=xgb.DMatrix(xtrain,label=ytrain)

d_valid=xgb.DMatrix(xtest,label=ytest)

num_round = 200

model = xgb.train(param3, d_train, num_round)

# make prediction

y_pred = model.predict(d_valid)

print(time()-time0)

print(np.sqrt(mean_squared_error(ytest,y_pred)))

plt.plot(y_pred)

plt.plot(ytest)

plt.legend(['y_pred','y_test'])

plt.show()
time0=time()

param1 = {'silent':True 

          ,'obj':'reg:linear' 

          ,"subsample":1

          ,"max_depth":6

          ,"eta":0.3

          ,"gamma":0

          ,"lambda":1

          ,"alpha":0

          ,"colsample_bytree":1

          ,"colsample_bylevel":1

          ,"colsample_bynode":1

          ,"nfold":5}

d_train=xgb.DMatrix(xtrain,label=ytrain)

d_valid=xgb.DMatrix(xtest,label=ytest)

num_round = 200

model = xgb.train(param1, d_train, num_round)

# make prediction

y_pred = model.predict(d_valid)

print(time()-time0)

print(np.sqrt(mean_squared_error(ytest,y_pred)))

plt.plot(y_pred)

plt.plot(ytest)

plt.legend(['y_pred','y_test'])

plt.show()
# using XGB to predict binary result

import numpy as np

import xgboost as xgb

import matplotlib.pyplot as plt

from xgboost import XGBClassifier as XGBC

from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split as TTS

from sklearn.metrics import confusion_matrix as cm, recall_score as recall, roc_auc_score as auc,accuracy_score as accuracy
class_1 = 500 

class_2 = 50 

centers = [[0.0, 0.0], [2.0, 2.0]] 

clusters_std = [1.5, 0.5] 

X, y = make_blobs(n_samples=[class_1, class_2],

                  centers=centers,

                  cluster_std=clusters_std,

                  random_state=0, shuffle=False)
Xtrain, Xtest, Ytrain, Ytest = TTS(X,y,test_size=0.3,random_state=420)
(y == 1).sum() / y.shape[0]
clf = XGBC().fit(Xtrain,Ytrain)

ypred = clf.predict(Xtest)

clf.score(Xtest,Ytest)
cm(Ytest,ypred,labels=[1,0])
recall(Ytest,ypred)
auc(Ytest,clf.predict_proba(Xtest)[:,1])
clf_ = XGBC(scale_pos_weight=10).fit(Xtrain,Ytrain) 

ypred_ = clf_.predict(Xtest) 

clf_.score(Xtest,Ytest)
cm(Ytest,ypred_,labels=[1,0])
recall(Ytest,ypred_)
auc(Ytest,clf_.predict_proba(Xtest)[:,1])
for i in [1,5,10,20,30]:

    clf_ = XGBC(scale_pos_weight=i).fit(Xtrain,Ytrain)

    ypred_ = clf_.predict(Xtest)

    print(i)

    print("\tAccuracy:{}".format(clf_.score(Xtest,Ytest)))

    print("\tRecall:{}".format(recall(Ytest,ypred_)))

    print("\tAUC:{}".format(auc(Ytest,clf_.predict_proba(Xtest)[:,1])))
dtrain = xgb.DMatrix(Xtrain,Ytrain)

dtest = xgb.DMatrix(Xtest,Ytest)
param= {'silent':True,'objective':'binary:logistic',"eta":0.1,"scale_pos_weight":1} 

num_round = 100

bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtest)
ypred = preds.copy() 

ypred[preds > 0.5] = 1 

ypred[ypred != 1] = 0
scale_pos_weight = [1,5,10]

names = ["negative vs positive: 1"

         ,"negative vs positive: 5"

         ,"negative vs positive: 10"]
for name,i in zip(names,scale_pos_weight):

    param= {'silent':True,'objective':'binary:logistic'

            ,"eta":0.1,"scale_pos_weight":i}

    clf = xgb.train(param, dtrain, num_round)

    preds = clf.predict(dtest)

    ypred = preds.copy()

    ypred[preds > 0.5] = 1

    ypred[ypred != 1] = 0

    print(name)

    print("\tAccuracy:{}".format(accuracy(Ytest,ypred)))

    print("\tRecall:{}".format(recall(Ytest,ypred)))

    print("\tAUC:{}".format(auc(Ytest,preds)))
for name,i in zip(names,scale_pos_weight):

    for thres in [0.3,0.5,0.7,0.9]:

        param= {'silent':True,'objective':'binary:logistic'

        ,"eta":0.1,"scale_pos_weight":i}

        clf = xgb.train(param, dtrain, num_round)

        preds = clf.predict(dtest)

        ypred = preds.copy()

        ypred[preds > thres] = 1

        ypred[ypred != 1] = 0

        print("{},thresholds:{}".format(name,thres))

        print("\tAccuracy:{}".format(accuracy(Ytest,ypred)))

        print("\tRecall:{}".format(recall(Ytest,ypred)))

        print("\tAUC:{}".format(auc(Ytest,preds)))