# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

filename='../input/titanic/train.csv'

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv(filename)

df
df.info()
df.drop('Cabin',axis=1,inplace=True)

df['Age']=df['Age'].fillna(df['Age'].mean())
import matplotlib.pyplot as plt

age_dict={'<25':0,'<50':0,'>50':0}

for i in df.Age:

    if i<=25:

        age_dict['<25']+=1

    elif i<50:

        age_dict['<50']+=1

    else :

        age_dict['>50']+=1

plt.bar(*zip(*age_dict.items()));



        
df.Parch.value_counts()
df.groupby('Parch').Survived.value_counts()

Parcher={0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six'}

df['Parch']=df['Parch'].map(Parcher)

df
df.groupby('Embarked').Survived.value_counts()
df.groupby('Sex').Survived.value_counts()
new_df=pd.get_dummies(df[['Sex','Parch','Embarked']])


new_df
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix
tt=pd.read_csv('../input/titanic/test.csv')


Parcher={0:'Zero',1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six'}

tt['Parch']=tt['Parch'].map(Parcher)

new_tt=pd.get_dummies(tt[['Sex','Parch','Embarked']])

len(new_tt)
dtc=DecisionTreeClassifier()

X=new_df

y=df.Survived

print(X,y)
X
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
dtc.fit(X_train,y_train)

ypred=dtc.predict(X_test)

print(accuracy_score(y_test,ypred))

print(confusion_matrix(y_test,ypred))
from sklearn.ensemble import RandomForestClassifier
max=0

count=0

for i in range(10,201,10):

    rfc=RandomForestClassifier(n_estimators=i)

    rfc.fit(X_train,y_train)

    ypred=rfc.predict(X_test)

    ans=accuracy_score(y_test,ypred)

    if ans>max:

        max=ans

        count=i

print(max,count)


from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



max=0

count=0

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')

    ans=np.mean(scores)

    if ans>max:

        max=ans

        count=i

print(max,count)

max=0

count=0

for i in range(1,30):

    dtc=DecisionTreeClassifier(max_depth=i)

    scores=cross_val_score(dtc,X,y,cv=10,scoring='accuracy')

    ans=np.mean(scores)

    if ans>max:

        max=ans

        count=i

print(max,count)
max=0

count=0

for i in range(10,201,10):

    dtc=RandomForestClassifier(n_estimators=i)

    scores=cross_val_score(dtc,X,y,cv=10,scoring='accuracy')

    ans=np.mean(scores)

    if ans>max:

        max=ans

        count=i

print(max,count)
new_tt
dtc=RandomForestClassifier(n_estimators=10)

dtc.fit(X,y)

answer=dtc.predict(new_tt)
from collections import Counter
num=Counter(answer)

print(num)
tt.PassengerId
len(new_tt)

final=pd.DataFrame(answer,tt.PassengerId)

final.rename(columns={0:'Survived'},inplace=True)

final.reset_index(inplace=True)
final.to_csv('final.csv',index=False)