import pandas as pd

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score

d=pd.read_csv('../input/mushroom-classification/mushrooms.csv')

le=LabelEncoder()

for i in d:

    le.fit(d[i])

    d[i]=le.transform(d[i])

d.head()
l=d.corr()['class'] 

x=[]

for i in l.index:

    if (l[i]>0.3 and l[i]<1) or (l[i]<-0.3):

        x.append(i)

X=d[x]

print(X.columns)

y=d['class']

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33,random_state=5)
lr = LogisticRegression(random_state=0).fit(X, y)

clf=MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(Xtrain, ytrain)

lr.fit(Xtrain,ytrain)

lrpred=lr.predict(Xtest)

ypred=pd.DataFrame(clf.predict(Xtest))

cnf=confusion_matrix(ytest,ypred)

cnf 

xg=XGBClassifier()

xg.fit(Xtrain,ytrain)

xgpred=xg.predict(Xtest)
print(accuracy_score(ytest,ypred))

print(precision_score(ytest,ypred))

print(accuracy_score(ytest,xgpred))

print (precision_score(ytest,xgpred))

print(accuracy_score(ytest,lrpred))
print(ytest.head(15))

print(ypred.head(15))