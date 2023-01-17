# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # non-linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X=pd.read_csv('../input/train.csv')

X_test=pd.read_csv('../input/test.csv')

X.dropna(axis=0,subset=['Survived'],inplace=True)

X.drop('Embarked',axis=1,inplace=True)

#X_test1=X_test.drop('Embarked',axis=1)

#print(X.columns)

a=X.isnull().sum()

print(a[a>0])

b=X_test.isnull().sum()

print(b[b>0])

print(X.shape)

print(X_test.shape)
X.drop(['Cabin'],axis=1,inplace=True)

X_test.drop(['Cabin'],axis=1,inplace=True)

#print(b['Fare'])

X_test[X_test['Fare'].isnull()]
c=X_test.groupby(['Pclass','Embarked']).mean()['Fare']

print(c)

X_test['Fare'] = X_test.groupby(['Pclass','Embarked'])['Fare'].apply(lambda x: x.fillna(x.mean()))

X_test.iloc[152,:]
X_test.drop('Embarked',axis=1,inplace=True)
X['Title'] = X['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

X_test['Title'] = X_test['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

X['Title'].unique()


#replacing all titles with mr, mrs, miss, master

def replace_titles(x):

    title=x['Title']

    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:

        return 'Mr'

    elif title in ['Mme','the Countess','Dona']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='Male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

X['Title']=X.apply(replace_titles, axis=1)

X_test['Title']=X_test.apply(replace_titles, axis=1)
print(X['Title'].unique())

print(X_test['Title'].unique())
X['Age']=X.groupby(['Title'])['Age'].apply(lambda x: x.fillna(x.mean()))

X_test['Age']=X_test.groupby(['Title'])['Age'].apply(lambda x: x.fillna(x.mean()))
X.drop('Name',axis=1,inplace=True)

X_test.drop('Name',axis=1,inplace=True)
X['Family_Size']=X['SibSp']+X['Parch']+1

X_test['Family_Size']=X_test['SibSp']+X['Parch']+1
x=X['Family_Size'].unique()

x.sort()

z=X['Family_Size'].value_counts()

y1=[]

j=0

for i in x:

    a=X[X['Family_Size']==i]

    y1.append( a[a['Survived']==1].shape[0])

print(y1)



import matplotlib.pyplot as plt   

%matplotlib inline

plt.plot(x,y1)

plt.show()
y=X['Survived']

X.drop('Survived',axis=1,inplace=True)
X['Age*Pclass']=X['Age']*X['Pclass']

X_test['Age*Pclass']=X_test['Age']*X_test['Pclass']

X['Fare_per_person']=X['Fare']/X['Family_Size']

X_test['Fare_per_person']=X_test['Fare']/X_test['Family_Size']
A=pd.get_dummies(X)

B=pd.get_dummies(X_test)

print(A.shape)

print(B.shape)

A,B=A.align(B,join='inner',axis=1)

print(A.shape)

print(B.shape)
u=[]

for col in A.columns:

    if A[col].isnull().any():

        u.append(col)

print(len(u))
A.drop(u,axis=1,inplace=True)

B.drop(u,axis=1,inplace=True)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

model1=RandomForestClassifier(n_estimators=100,random_state=0)

model2=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)

model3=RandomForestClassifier(n_estimators=200,random_state=0,max_depth=7)

model4=RandomForestClassifier(n_estimators=300,random_state=0)

model5=RandomForestClassifier(n_estimators=200,random_state=0)

model6=RandomForestClassifier(n_estimators=200,max_leaf_nodes=1000,random_state=0)

model8=RandomForestClassifier(n_estimators=75,random_state=0)

model9=RandomForestClassifier(n_estimators=200,min_samples_split=0.2,random_state=0)

model10=RandomForestClassifier(n_estimators=200,min_samples_split=6,min_samples_leaf=6,

                               max_leaf_nodes=1000,random_state=0)

a=[]

from sklearn.model_selection import cross_val_score

scores1 = cross_val_score(model1,A,y,cv=5,scoring='accuracy')

q=scores1.mean()

print(q)

a.append(q)
scores3 = cross_val_score(model3,A,y,cv=5,scoring='accuracy')

q=scores3.mean()

print(q)

a.append(q)
scores2 = cross_val_score(model2,A,y,cv=5,scoring='accuracy')

q=scores2.mean()

print(q)

a.insert(1,q)
scores4 = cross_val_score(model4,A,y,cv=5,scoring='accuracy')

q=scores4.mean()

print(q)

a.append(q)
scores5 = cross_val_score(model5,A,y,cv=5,scoring='accuracy')

q=scores5.mean()

print(q)

a.append(q)
scores6 = cross_val_score(model6,A,y,cv=5,scoring='accuracy')

q=scores6.mean()

print(q)

a.append(q)
from xgboost import XGBClassifier

model7=XGBClassifier()
scores7 = cross_val_score(model7,A,y,cv=5,scoring='accuracy')

q=scores7.mean()

print(q)

a.append(q)
scores8 = cross_val_score(model8,A,y,cv=5,scoring='accuracy')

q=scores8.mean()

print(q)

a.append(q)
scores9 = cross_val_score(model9,A,y,cv=5,scoring='accuracy')

q=scores9.mean()

print(q)

a.append(q)
scores10 = cross_val_score(model10,A,y,cv=5,scoring='accuracy')

q=scores10.mean()

print(q)

a.append(q)
b=B.isnull().sum()

b[b>0]
model6.fit(A,y)

preds=model6.predict(B)

output=pd.DataFrame({'PassengerId':B.index + 892,'Survived':preds})

output.to_csv('submission.csv',index=False)

#output.head()