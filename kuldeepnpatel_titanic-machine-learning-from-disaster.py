# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_set= pd.read_csv("../input/train.csv")
#test_set= pd.read_csv("../input/test.csv")
train_set.shape
data=train_set.copy()
#frames = [train_set,test_set]
#data = pd.concat(frames,sort=False)
#data.shape
data
data.head()
data.info()
data.describe(include="all")
data.drop(["PassengerId","Ticket","Fare","Cabin"],axis=1,inplace=True)
data.head()
mean1=int(data[data.Pclass==1]["Age"].mean())
print("mean of 1st class",mean1)
mean2=int(data[data.Pclass==2]["Age"].mean())
print("mean of 2nd class",mean2)
mean3=int(data[data.Pclass==3]["Age"].mean())
print("mean of 3rd class",mean3)
def impute_age(a):
    Age= a[0]
    Pclass=  a[1]
    
    if pd.isnull(Age):
        if Pclass==1:
            return mean1
        elif Pclass==2:
            return mean2
        elif Pclass ==3:
            return mean3
    else:
        return data.Age
data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)
data.Age.isnull().sum()
#filling missing Age with interpolate
#data.Age.interpolate(inplace=True)
#data.Age=data.Age.astype(int)
#filling nan of column Embarked with S
data.Embarked.fillna(data.Embarked.value_counts().index[0],inplace=True)
#Embarked is filled
data.Embarked.isnull().sum()
data.info()
#sibsp+parch is alone or with family
data["family_or_alone"]= data.SibSp + data.Parch
data["family_type"]= ["alone" if x==0 else "small_family" if x<5 else "big_family" for x in data.family_or_alone ]
data.head()
sns.countplot(x="Survived",data=data,hue="family_type")
plt.show()
#converting age to children, young people,old people
data["age_type"]=["children" if x<17  else "adult" if x<81 else "none" for x in data.Age]
        
sns.countplot(x="Survived",data=data,hue="age_type")
plt.show()
sns.countplot(x="Survived",data=data,hue="Pclass")
plt.show()
data.Survived.value_counts()
data.Survived.value_counts(normalize=True)
sns.countplot(x="Survived",data=data,hue="Sex")
plt.show()
sns.countplot(x="Survived",data=data,hue="family_or_alone")
plt.show()
sns.countplot(x="Embarked",data=data,hue="Survived")
data.Pclass.value_counts()
pd.crosstab(data.Survived,data.Pclass,margins=True)
data.drop(["Name","family_or_alone","family_type","age_type"],axis=1,inplace=True)
data=pd.get_dummies(data,drop_first=True)
data.head()
X=data.drop("Survived",axis=1)
y=data.Survived
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))