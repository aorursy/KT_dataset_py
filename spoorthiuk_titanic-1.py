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



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/titanic/train.csv")

df.head(10)
df.info()
import seaborn as sns

sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt='.3f')
def what (passenger):

    age,sex=passenger

    if age<16:

        return "child"

    else:

        return dict(male="man",female="woman")[sex]

df['Who']=df[['Age','Sex']].apply(what,axis=1)

        
df.head(10)
df['Deck']=df.Cabin.str[0]

df['Alone']=~(df.SibSp+df.Parch).astype(bool)
df.head(10)
sns.factorplot('Pclass','Survived',data=df)
sns.factorplot('Pclass','Survived',data=df,hue='Who')
sns.factorplot('Alone','Survived',data=df,hue='Sex',col='Pclass')
sns.barplot('Deck','Survived',data=df,order=['A','B','C','D','E','F','G'])
dk={'male':0,'female':1}

df['Sex']=df.Sex.map(dk)

dk={'S':3,'Q':2,'C':1}

df['Embarked']=df.Embarked.map(dk)

dk={'child':3,'woman':2,'man':1}

df['Who']=df.Who.map(dk)

dk={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}

df['Deck']=df.Deck.map(dk)
df.head()
df.isna().sum()

df['Age']=df['Age'].fillna(df['Age'].mean())

df['Deck']=df['Deck'].fillna(0)

df['Embarked'].value_counts()
df['Embarked']=df['Embarked'].fillna(3)

def fam_size(par):

    x,y=par

    s=x+y+1

    if(1==s):

        return 1

    elif(2<=s<=4):

        return 2

    else:

        return 3

df['Fam_size']=df[['SibSp','Parch']].apply(fam_size,axis=1)
Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}

df['Titles'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

df.head()
#one hot encoding

df['Titles']=df.Titles.map(Title_Dictionary)

df.head()

title_dummies=pd.get_dummies(df['Titles'],prefix='Title')

df=pd.concat([df,title_dummies],axis=1)
def p_class(par):

    w,p=par

    if p==1:

        if w==1:

            return 1

        elif w==2:

            return 2

        else :

            return 3

    elif p==2:

        if w==1:

            return 4

        elif w==2:

            return 5

        else :

            return 6

    else:

        if w==1:

            return 7

        elif w==2:

            return 8

        else :

            return 9

df['P_W']=df[['Who','Pclass']].apply(p_class,axis=1)

df.head()
df.drop(['PassengerId','Name','Ticket','Alone','Titles'],axis=1)
X_train=df.drop(['Survived','PassengerId','Name','Ticket','Alone','Titles','Cabin'],axis=1)

Y_train=df['Survived']

X_train.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3)

x_train.head()
from sklearn.ensemble import RandomForestClassifier

y_train.head()

rf = RandomForestClassifier(criterion = "gini", 

                                       min_samples_leaf = 3, 

                                       min_samples_split = 10,   

                                       n_estimators=100, 

                                       max_features=0.5, 

                                       oob_score=True, 

                                       random_state=1, 

                                       n_jobs=-1)

rf = rf.fit(x_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score

act = accuracy_score(y_train,rf.predict(x_train))

print('Training Accuracy is: ',(act*100))

p = precision_score(y_train,rf.predict(x_train))

print('Training Precision is: ',(p*100))

r = recall_score(y_train,rf.predict(x_train))

print('Training Recall is: ',(r*100))

f = f1_score(y_train,rf.predict(x_train))

print('Training F1 Score is: ',(f*100))
act = accuracy_score(y_test,rf.predict(x_test))

print('Test Accuracy is: ',(act*100))

p = precision_score(y_test,rf.predict(x_test))

print('Test Precision is: ',(p*100))

r = recall_score(y_test,rf.predict(x_test))

print('Test Recall is: ',(r*100))

f = f1_score(y_test,rf.predict(x_test))

print('Test F1 Score is: ',(f*100))
df_t=pd.read_csv("/kaggle/input/titanic-mine/test.csv")

df_t.head(10)
def what (passenger):

    age,sex=passenger

    if age<16:

        return "child"

    else:

        return dict(male="man",female="woman")[sex]

df_t['Who']=df_t[['Age','Sex']].apply(what,axis=1)

df_t.head()
df_t['Deck']=df_t.Cabin.str[0]

df_t['Alone']=~(df_t.SibSp+df.Parch).astype(bool)
dk={'male':0,'female':1}

df_t['Sex']=df_t.Sex.map(dk)

dk={'S':3,'Q':2,'C':1}

df_t['Embarked']=df_t.Embarked.map(dk)

dk={'child':3,'woman':2,'man':1}

df_t['Who']=df_t.Who.map(dk)

dk={'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}

df_t['Deck']=df_t.Deck.map(dk)

df_t.head()
df_t.isna().sum()

df_t['Age']=df_t['Age'].fillna(df['Age'].mean())

df_t['Deck']=df_t['Deck'].fillna(0)

df_t['Embarked'].value_counts()
df_t['Embarked']=df_t['Embarked'].fillna(3)
def fam_size(par):

    x,y=par

    s=x+y+1

    if(1==s):

        return 1

    elif(2<=s<=4):

        return 2

    else:

        return 3

df_t['Fam_size']=df_t[['SibSp','Parch']].apply(fam_size,axis=1)
Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}

df_t['Titles'] = df_t['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

df_t.head()
#one hot encoding

df_t['Titles']=df_t.Titles.map(Title_Dictionary)

df_t.head()

title_dummies=pd.get_dummies(df_t['Titles'],prefix='Title')

df_t=pd.concat([df_t,title_dummies],axis=1)
def p_class(par):

    w,p=par

    if p==1:

        if w==1:

            return 1

        elif w==2:

            return 2

        else :

            return 3

    elif p==2:

        if w==1:

            return 4

        elif w==2:

            return 5

        else :

            return 6

    else:

        if w==1:

            return 7

        elif w==2:

            return 8

        else :

            return 9

df_t['P_W']=df_t[['Who','Pclass']].apply(p_class,axis=1)

df_t.head()
X_train=df_t.drop(['PassengerId','Name','Ticket','Titles','Cabin'],axis=1)

X_train.insert(15,'Title_Royalty',0)
X_train.head()

X_train['Fare']=X_train['Fare'].fillna(X_train['Fare'].mean())

x_train.shape

x_train.head()
X_train=X_train.drop('Alone',axis=1)
res=rf.predict(X_train)

print(res)
sub=pd.read_csv("/kaggle/input/titanic-mine/gender_submission.csv")

sub.shape
sub['PassengerId']=df_t['PassengerId']
sub['Survived']=res
sub.head()
sub.to_csv("Submission1.csv",index=False)