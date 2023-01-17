import numpy as np
import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train=pd.read_csv("../input/train.csv")
train.head()
train.info()

train.describe()
sns.heatmap(train.isnull(), cbar=False)
sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
dis=sns.distplot(train['Age'].dropna(),bins=30,kde=False)

dis

#kde removes the line 
sns.countplot(x='SibSp',data=train)
bs=train.iloc[:,9].values
plt.hist(bs,bins=20,rwidth=1)

plt.show()
sns.boxplot(x="Pclass",y="Age",data=train)
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

        

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis= 1) # hint apply function in age column
sns.heatmap(train.isnull(), cbar=False)
train=train.drop(['Cabin'],axis=1)             # drop cabin columns
train.head()
sns.heatmap(train.isnull())
train.dropna()
sns.heatmap(train.isnull(), cbar=False)
a=pd.get_dummies(train.Sex, prefix='Sex').loc[:, 'Sex_male':]
a.head()
e=pd.get_dummies(train.Embarked).iloc[:, 1:]
e.head()
train = pd.concat([train,a,e],axis=1)

train.head(2)
train.head()
train=train.drop(['Sex','Name','Embarked','Ticket'],axis=1)
train.head()
train=train.drop(columns=('PassengerId'))
train.head()
x=train.drop('Survived',axis=1).values
y=train['Survived'].values
z=train.nunique()

z
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
x_train
x_test
y_train
train.dropna(inplace=True)
from sklearn.linear_model import LogisticRegression

log_model=LogisticRegression()

log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
sns.heatmap(cm,annot=True)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))