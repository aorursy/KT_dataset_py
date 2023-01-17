

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_test = pd.read_csv("../input/titanic/test.csv")

df_train = pd.read_csv('../input/titanic/train.csv')
sns.countplot(x='Survived',hue='Sex',data=df_train)
sns.countplot(x='Survived',hue='Pclass',data=df_train)
sns.distplot(df_train['Age'].dropna(),bins=30,kde=False)
df_train.info()
df_train['Fare'].hist(bins=100,figsize=(10,10))
df_train[df_train['Fare']>500]
#adjusting for inflation, one dollar back then is equal to $25.89/

#lets ajust for inflation
df_train[df_train['Fare']>500]['Fare']*25.89
age_means = pd.pivot_table(df_train,values = 'Age',index= 'Pclass',aggfunc='mean')

age_means
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        

        if Pclass ==1:

            return 38

        elif Pclass ==2:

            return 30

        else:

            return 25

    else:

        return Age
df_train['Age'] = df_train[['Age','Pclass']].apply(impute_age,axis=1)
df_train.drop('Cabin',axis=1,inplace=True)
sex = pd.get_dummies(df_train['Sex'],drop_first=True)
embark = pd.get_dummies(df_train['Embarked'],drop_first=True)
embark.head()


train = pd.concat([df_train,sex,embark],axis=1)
train.head()
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train.drop(['PassengerId'],axis=1,inplace=True)
X = train.drop('Survived',axis=1)

y = train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
predictions
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score
confusion_matrix(y_test,predictions)
precision_score(y_test,predictions)
pd.DataFrame(index= list(logmodel.coef_),data = list(X_train.columns))
logmodel.intercept_
gma  = pd.read_csv("../input/grandmas-attributes/grandma.csv")
gma.head()
gma_surv = logmodel.predict(gma.drop('Adj. Fare',axis=1))
print(gma_surv)
logmodel.predict_proba(gma.drop('Adj. Fare',axis=1))