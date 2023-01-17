# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/titanic/train.csv')
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
df.drop(['PassengerId','Name'],axis=1,inplace=True)
df.head(10)
df.shape
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['sex_n']=le.fit_transform(df.Sex)    #male=1,female=0
df.drop('Sex',axis=1,inplace=True)
df.head() 
len(df.Ticket.unique()) # on checking unique no oftickets is almost 70% of total passengers so its almost like passenger id or name
                        # therfore ticket has no impact on chances of survival,therefore column ticket is dropped
df.drop('Ticket',axis=1,inplace=True)
df.head()
pd.crosstab(df.Pclass,df.Survived).plot(kind='bar')
pd.crosstab(df.SibSp,df.Survived).plot(kind='bar')
pd.crosstab(df.Parch,df.Survived).plot(kind='bar')
pd.crosstab(df.Fare,df.Survived).plot(kind='bar')
df.Age.isnull().sum()
df.Age.fillna(df.Age.mode(),inplace=True)
pd.crosstab(df.Age,df.Survived).plot(kind='bar')
df.drop('Cabin',axis=1,inplace=True)
df1=pd.get_dummies(df.Embarked)
df1.head()
df2=pd.concat([df,df1],axis=1)
df2.head() 
df2.drop(['Embarked','Cabin','Q'],axis=1,inplace=True)
df2.head()
from sklearn.linear_model import LogisticRegression
lsr=LogisticRegression(max_iter=500,C=2.0)
from sklearn.ensemble import RandomForestClassifier
rdm_forest=RandomForestClassifier(n_estimators=300)
from sklearn.svm import SVC
svc=SVC()
from sklearn.naive_bayes import MultinomialNB
naive_clf=MultinomialNB()
from sklearn.tree import DecisionTreeClassifier
dec_tree_clf=DecisionTreeClassifier(criterion='entropy')
from sklearn.model_selection import cross_val_score
x=df2.drop('Survived',axis=1)
y=df2.Survived
cross_val_score(lsr,x,y)
cross_val_score(svc,x,y)
cross_val_score(rdm_forest,x,y)
cross_val_score(naive_clf,x,y)
cross_val_score(dec_tree_clf,x,y)
from sklearn.ensemble import GradientBoostingClassifier
grad_boost_clf=GradientBoostingClassifier(n_estimators=200)
cross_val_score(grad_boost_clf,x,y)
clf=grad_boost_clf.fit(x,y)
test_data.head()
test_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
test_data['sex_n']=le.fit_transform(test_data.Sex)    #male=1,female=0
test_data.drop('Sex',axis=1,inplace=True)
test_data.head() 
df4=pd.get_dummies(test_data.Embarked)
test_data_final=pd.concat([test_data,df4],axis=1)
test_data_final.drop('Q',axis=1,inplace=True)
test_data_final.head()
test_data_final.drop('Embarked',axis=1,inplace=True)
test_data_final.head()

test_data_final.Age.fillna(test_data_final.Age.mean(),inplace=True)
test_data_final.Fare.fillna(test_data_final.Fare.mean(),inplace=True)
test_data_final.head()
submission['Survived']=clf.predict(test_data_final)
submission.head()

my_submission=submission[['PassengerId','Survived']]
my_submission.to_csv('submission2_final.csv', index=False)
