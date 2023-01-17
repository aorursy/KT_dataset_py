# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
Data=pd.concat([Data,pd.get_dummies(Data.Sex)],axis='columns')
Data=Data.drop(['PassengerId', 'Name', 'Embarked','Ticket','Cabin','Sex'],'columns')

Data.shape

Data=Data.drop_duplicates()

Data.shape

Data['Age'] = Data['Age'].fillna(Data['Age'].mean())
Data.head(1)
plt.subplot(2,6,1)

plt.hist(Data[Data.Survived==1].values[:,1])

plt.subplot(2,6,7)

plt.hist(Data[Data.Survived==0].values[:,1])



plt.subplot(2,6,2)

plt.hist(Data[Data.Survived==1].values[:,2])

plt.subplot(2,6,8)

plt.hist(Data[Data.Survived==0].values[:,2])



plt.subplot(2,6,3)

plt.hist(Data[Data.Survived==1].values[:,3])

plt.subplot(2,6,9)

plt.hist(Data[Data.Survived==0].values[:,3])



plt.subplot(2,6,4)

plt.hist(Data[Data.Survived==1].values[:,4])

plt.subplot(2,6,10)

plt.hist(Data[Data.Survived==0].values[:,4])



plt.subplot(2,6,5)

plt.hist(Data[Data.Survived==1].values[:,5])

plt.subplot(2,6,11)

plt.hist(Data[Data.Survived==0].values[:,5])



plt.subplot(2,6,6)

plt.hist(Data[Data.Survived==1].values[:,6])

plt.subplot(2,6,12)

plt.hist(Data[Data.Survived==0].values[:,6])
X=Data.values[:,1:8]

y=Data.values[:,0]
from sklearn import *
X=preprocessing.MinMaxScaler().fit_transform(X)

X=preprocessing.PolynomialFeatures(2).fit_transform(X)
model1=linear_model.SGDClassifier(alpha=0.00701,random_state=64,n_iter=4,n_jobs=-1)

model2=linear_model.LogisticRegression(C=9.1,max_iter=100,verbose=0,n_jobs=-1)

model3=svm.SVC(coef0=0.9,degree=6,random_state=0,kernel='poly')

model4=neighbors.KNeighborsClassifier(leaf_size=3,n_neighbors=4,p=2,n_jobs=-1)

model5=tree.DecisionTreeClassifier(max_depth=2,min_samples_leaf=0.1,min_samples_split=0.1,random_state=0)

model6=ensemble.RandomForestClassifier(n_estimators=66,max_depth=2,min_samples_leaf=0.1,min_samples_split=0.1,random_state=0,n_jobs=-1)
Xnew=X

Xnew.shape
model1.fit(X,y)

model2.fit(X,y)

model3.fit(X,y)

model4.fit(X,y)

model5.fit(X,y)

model6.fit(X,y)

Xnew=np.append(Xnew,model1.predict(X).reshape(777,1), axis=1)

Xnew=np.append(Xnew,model2.predict(X).reshape(777,1), axis=1)

Xnew=np.append(Xnew,model3.predict(X).reshape(777,1), axis=1)

Xnew=np.append(Xnew,model4.predict(X).reshape(777,1), axis=1)

Xnew=np.append(Xnew,model5.predict(X).reshape(777,1), axis=1)

Xnew=np.append(Xnew,model6.predict(X).reshape(777,1), axis=1)

Xnew.shape
final=neural_network.MLPClassifier(hidden_layer_sizes=(100,50,25),max_iter=450,random_state=95,beta_1=0.90000000000000002,beta_2=0.70000000000000007)
final.fit(Xnew,y)
t=pd.concat([test,pd.get_dummies(test.Sex)],axis='columns')
t=t.drop(['PassengerId', 'Name', 'Embarked','Ticket','Cabin','Sex'],'columns')

t['Age'] = t['Age'].fillna(t['Age'].mean())

t.head(1)
t=t.fillna(0)

Xne=t.values

Xne=preprocessing.MinMaxScaler().fit_transform(Xne)

Xne=preprocessing.PolynomialFeatures(2).fit_transform(Xne)

Xnew=Xne

t.shape
Xnew=np.append(Xnew,model1.predict(Xne).reshape(418,1), axis=1)

Xnew=np.append(Xnew,model2.predict(Xne).reshape(418,1), axis=1)

Xnew=np.append(Xnew,model3.predict(Xne).reshape(418,1), axis=1)

Xnew=np.append(Xnew,model4.predict(Xne).reshape(418,1), axis=1)

Xnew=np.append(Xnew,model5.predict(Xne).reshape(418,1), axis=1)

Xnew=np.append(Xnew,model6.predict(Xne).reshape(418,1), axis=1)
Ypred=final.predict(Xnew)

Ypred=Ypred.astype(int)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':Ypred})

submission.head(100)
filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)



print('Saved file: ' + filename)