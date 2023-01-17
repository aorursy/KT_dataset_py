# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Graphs

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/heart.csv")
df.columns

df.shape
df.head(10)
df.dtypes
df.describe()
df.isnull().sum()
df.hist(figsize=(12,12))
df.plot(kind='box',subplots=True,layout=(4,4),sharex=False,sharey=False,figsize=(18,18))
df.corr()
fig=plt.figure(figsize=(15,15))

ax=fig.add_subplot(111)

cax=ax.matshow(df.corr(),vmin=-1,vmax=1)

fig.colorbar(cax)

ticks=np.arange(0,14,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)
df.groupby('target').size()
p_risk = (len(df.loc[(df['target']==1) ])/len(df.loc[df['target']]))*100

print("Percentage of people at risk : ", p_risk)
abc = pd.crosstab(df['sex'],df['target'])

abc
female_risk_percent = (len(df.loc[((df['sex']==0) & df['target']==1) ])/len(df.loc[df['sex']==0]))*100

male_risk_percent = (len(df.loc[((df['sex']==1) & df['target']==1) ])/len(df.loc[df['sex']==1]))*100

print('percentage males at risk : ',male_risk_percent)

print('percentage females at risk : ',female_risk_percent)
abc.plot(kind='bar', stacked=False, color=['#f5b7b1','#a9cce3'])
xyz = pd.crosstab(df.age,df.target)

xyz.plot(kind='bar',stacked=False,figsize=(15,8))
pqr = pd.crosstab(df.cp,df.target)

pqr
pqr.plot(kind='bar',figsize=(12,5))
mno = pd.crosstab(df.thal,df.target)

mno
mno.plot(kind='bar', stacked=False, color=['#2471a3','#ec7063'],figsize=(12,5))
array = df.values

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

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC
models=[]

models.append(('LR  :', LogisticRegression()))

models.append(('LDA :', LinearDiscriminantAnalysis()))

models.append(('KNN :', KNeighborsClassifier()))

models.append(('CART:', DecisionTreeClassifier()))

models.append(('NB  :', GaussianNB()))

models.append(('SVM :', SVC()))

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

    
qwerty =['LR', 'LDA', 'KNN', 'CART', 'NB', 'SVM'] 



fig = plt.figure(figsize=(10,10))

fig.suptitle("Algorithm Comparision")

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(qwerty)

plt.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
LR = LogisticRegression()

LR.fit(X_train, y_train)

predictions = LR.predict(X_validation)

print(accuracy_score(y_validation, predictions)*100)

print(classification_report(y_validation, predictions))