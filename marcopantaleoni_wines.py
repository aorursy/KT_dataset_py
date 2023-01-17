import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

plt.style.use('ggplot')
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

df = pd.read_csv(url, sep = ';')

df.columns
df.head()
fig, ax = plt.subplots(figsize=(12,17), nrows=4, ncols=3)

counter = 0

for column in df.columns:

  j = counter//4; i = counter%4;

  sns.distplot(df[column].values,ax=ax[i,j],rug=True)



  ax[i,j].set_title(column.capitalize())

  counter += 1



df.describe()
sns.barplot(x=df['quality'], y=df['alcohol'])
sns.regplot(x=df['quality'], y=df['alcohol'])
sns.kdeplot(df['quality'])
df.isnull().mean()
df['quality'].value_counts().unique
df['density'].value_counts(bins=5)
df.sort_values('quality',ascending=False)
df.dtypes
X= df.drop('quality', axis=1)

y=df['quality']
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=1)

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

Xtrain= scale.fit_transform(Xtrain)

Xtest= scale.fit_transform(Xtest)

from sklearn.ensemble import RandomForestClassifier

estimator=RandomForestClassifier(random_state=1)

estimator.fit(Xtrain,ytrain)

a=estimator.predict(Xtest)

accuracy_score(ytest,a)*100
from sklearn.metrics import classification_report

print(classification_report(ytest,a))
X= df.drop('quality', axis=1)

y=df['quality']

import statsmodels.api as sm

X2=sm.add_constant(X)

est=sm.OLS(y,X2)

est2= est.fit()

print(est2.summary())

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=1)

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()

Xtrain= scale.fit_transform(Xtrain)

Xtest= scale.fit_transform(Xtest)

from sklearn.tree import DecisionTreeClassifier

estimator=DecisionTreeClassifier(random_state=1)

estimator.fit(Xtrain,ytrain)

max_depth= estimator.tree_.max_depth

parameter_values=range(1,max_depth+1)

parameter_values
from statistics import mean

from sklearn.model_selection import cross_val_score

test_score=[]

s=[]

test=[]

train=[]



for par in parameter_values:

    estimator=DecisionTreeClassifier(criterion='entropy',max_depth=par,random_state=2)

    cr= cross_val_score(estimator, Xtrain,ytrain, cv=5)

    s.append(mean(cr))

    estimator.fit(Xtrain,ytrain)

    w=estimator.predict(Xtest)

    test.append(accuracy_score(ytest,w))

    p=estimator.predict(Xtrain)

    train.append(accuracy_score(ytrain,p))

plt.figure(figsize=(10,10))

plt.plot(s, label="score")

plt.plot(test,linewidth=4 ,label="test_acc")

plt.plot(train, label="train_acc")
np.argmax(test)
estimator=DecisionTreeClassifier(max_depth=13,random_state=2)

estimator.fit(Xtrain,ytrain)

e=estimator.predict(Xtest)

accuracy_score(ytest,e)*100
print(classification_report(ytest,e))

plt.figure(figsize=(15,15))

correlation=df.corr()

sns.heatmap(correlation,annot=True)
fig, ax= plt.subplots(figsize=(15,15), nrows=4, ncols=3)

counter=0

for column in df.columns:

   j = counter//4; i= counter%4;

   sns.distplot(df[column].values, ax=ax[i,j], rug=True)

   ax[i,j].set_title(column.capitalize())

   counter +=1  

   



from sklearn.svm import SVC

C_values=np.logspace(-3, 3, num=7, endpoint=True, base=10.0)

print(C_values)

test_accuracy=[]

train_accuracy=[]



for C_val in C_values:

  svm=SVC(kernel='linear', C=C_val)

  svm.fit(Xtrain,ytrain)

  train_accuracy.append(svm.score(Xtrain,ytrain))

  test_accuracy.append(svm.score(Xtest,ytest))



print(np.max(test_accuracy))





fig, ax = plt.subplots(figsize=(6,6))

ax.plot(np.log(C_values),train_accuracy,color='g',lw=2.,label='train_acc')

ax.plot(np.log(C_values),test_accuracy,color='r',lw=2.5,label='test_acc')

plt.title("Test and Train Accuracy versus Value of C")

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
d=svm.predict(Xtest)

print(classification_report(ytest,d))
svm = SVC(kernel='linear', C=1)

svm.fit(Xtrain,ytrain)



w = svm.coef_[0]

b = svm.intercept_[0]

print("VALUE OF INTERCEPT", b)

print("PERC. POSITIVE CLASS:", np.sum(ytrain==1)/len(ytrain))

print("PERC. NEGATIVE CLASS:", np.sum(ytrain==0)/len(ytrain))

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10))

var_idx = range(0,len(w))

ax.barh(var_idx, w, align='center')

ax.set_yticks(var_idx)

ax.set_yticklabels(df.columns[:-1])



plt.title("SVM Weights")

plt.show()

print(w)
from sklearn.model_selection import GridSearchCV

Parameter= {'C':[0.1,0.3,0.8,0.9,1.1,1.2,1.3,1.4],

           'kernel': ['linear', 'rbf'],

           'gamma':[0.1,0.3,0.8,0.9,1.1,1.2,1.3,1.4]}

grid_svc= GridSearchCV(estimator=svm,

             param_grid=Parameter,cv=9)
grid_svc.fit(Xtrain,ytrain)
grid_svc.best_params_
svc1=SVC(C= 1.2, gamma= 1.2, kernel= 'rbf')

svc1.fit(Xtrain,ytrain)

u=svc1.predict(Xtest)

print(classification_report(ytest,u))