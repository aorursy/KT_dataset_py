%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/heart.csv')

train.shape
train.head(10)
train.info()
train.hist(figsize=(12,12))
fig=plt.figure(figsize=(15,15))

ax=fig.add_subplot(111)

cax=ax.matshow(train.corr(),vmin=-1,vmax=1)

fig.colorbar(cax)

ticks=np.arange(0,14,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(train.columns)

ax.set_yticklabels(train.columns)
train.groupby('target').size()
p_risk = (len(train.loc[(train['target']==1) ])/len(train.loc[train['target']]))*100

print("Процентная доля лиц, подвергающихся риску заболеть : ", p_risk)
abc = pd.crosstab(train['sex'],train['target'])

abc
female_risk_percent = (len(train.loc[((train['sex']==0) & train['target']==1) ])/len(train.loc[train['sex']==0]))*100

male_risk_percent = (len(train.loc[((train['sex']==1) & train['target']==1) ])/len(train.loc[train['sex']==1]))*100

print('процент мужчин в группе риска : ',male_risk_percent)

print('процент женщин в группе риска : ',female_risk_percent)
abc.plot(kind='bar', stacked=False, color=['#f5b7b1','#a9cce3'])
xyz = pd.crosstab(train.age,train.target)

xyz.plot(kind='bar',stacked=False,figsize=(15,8))
pqr = pd.crosstab(train.cp,train.target)

pqr
pqr.plot(kind='bar',figsize=(12,5))
mno = pd.crosstab(train.thal,train.target)

mno
mno.plot(kind='bar', stacked=False, color=['#2471a3','#ec7063'],figsize=(12,5))
array = train.values

X = array[:, 0:13]

y = array[:, 13]



seed = 7

tsize = 0.2
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tsize, random_state=seed)
from sklearn.preprocessing import StandardScaler



scale = StandardScaler()

X_train_scale = scale.fit_transform(X_train)

X_train = pd.DataFrame(X_train_scale)

X_test_scale =scale.fit_transform(X_test)

X_test = pd.DataFrame(X_test_scale)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
models=[]

models.append(('LR  :', LogisticRegression()))

models.append(('CART:', DecisionTreeClassifier()))
results = []

names = []

score = 'accuracy'

seed = 7

folds = 10

X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size=0.2,random_state=seed)





for name, model in models:

    kfold = KFold(n_splits=folds,random_state=seed)

    cv_results = cross_val_score(model, X_train, y_train, scoring=score)

    results.append(cv_results)

    msg ="%s %f (%f)" % (name,cv_results.mean()*100,cv_results.std()*100)

    print(msg)
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
LR = LogisticRegression()

LR.fit(X_train, y_train)

predictions = LR.predict(X_validation)

print(accuracy_score(y_validation, predictions)*100)

print(classification_report(y_validation, predictions))
CART = DecisionTreeClassifier()

CART.fit(X_train, y_train)

predictions = CART.predict(X_validation)

print(accuracy_score(y_validation, predictions)*100)

print(classification_report(y_validation, predictions))



