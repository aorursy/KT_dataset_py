# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
trds=pd.read_csv("../input/titanic/train.csv")

teds=pd.read_csv("../input/titanic/test.csv")



trds.info()

teds.info()
teds.describe()
trds.head()
sns.barplot(x="Sex", y="Survived",data=trds)



print("Percentage of female survived:",trds["Survived"][trds["Sex"]=='female'].value_counts(normalize=True)[1]*100)



print("Percentage of male survived:",trds["Survived"][trds["Sex"]=='male'].value_counts(normalize=True)[1]*100)
trds=trds.set_index('PassengerId')

trds=trds.drop(["Cabin"],axis=1)

teds=teds.drop(["Cabin"],axis=1)
trds["Embarked"]=trds["Embarked"].fillna("S")

trds=trds.drop(["Name","Ticket"],axis=1)

teds=teds.drop(["Name","Ticket"],axis=1)
sex_mapping = {"male": 0, "female": 1}

trds['Sex'] = trds['Sex'].map(sex_mapping)

teds['Sex'] = teds['Sex'].map(sex_mapping)
em_mpng={"S":1,"C":2,"Q":3}

trds['Embarked']=trds['Embarked'].map(em_mpng)

teds['Embarked']=teds['Embarked'].map(em_mpng)
trds.head()
ctrds=trds.copy()
ctrds = ctrds.dropna()
ctrds['Age'] = ctrds['Age'].fillna(0).astype(np.int64)

ctrds['Fare']=ctrds['Fare'].fillna(0).astype(np.int64)
from sklearn.model_selection import train_test_split

X=ctrds.drop(['Survived'],axis=1)

y=ctrds[['Survived']]



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=22)
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier()

classifier.fit(X_train,y_train)

predict=classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(predict,y_test))

print(classification_report(predict,y_test))

accuracy_before=accuracy_score(predict,y_test)*100

print(accuracy_before)
from sklearn.neighbors import KNeighborsClassifier

c_kn=KNeighborsClassifier(n_neighbors=11)

c_kn.fit(X_train,y_train)

predict=c_kn.predict(X_test)

accuracy=accuracy_score(predict,y_test)*100

print(accuracy)
from sklearn.preprocessing import LabelEncoder

def encoding(feature):

    if (feature.dtype == 'object'):

        return LabelEncoder().fit_transform(feature)

    else:

        return feature


work_teds=teds.copy()

work_teds=work_teds.set_index('PassengerId')

#work_teds=work_teds.drop(['Name','Ticket','Cabin'],axis=1)

work_teds=work_teds.apply(encoding)
from sklearn.impute import SimpleImputer

impute=SimpleImputer(strategy='mean')

last_dataset=pd.DataFrame(impute.fit_transform(work_teds),columns=work_teds.columns)
survived=c_kn.predict(last_dataset)

submission=pd.DataFrame({'PassengerId':teds.PassengerId,'Survived':survived})
submission.to_csv('submission.csv',index=True)