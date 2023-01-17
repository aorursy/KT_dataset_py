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
training_data=pd.read_csv('../input/train.csv')

training_data.describe()
training_data.columns
training_data.info()
training_data.isnull().sum()
training_data['Age'].fillna(training_data['Age'].mode(),inplace=True)

training_data['Embarked'].fillna(training_data['Embarked'].mode(),inplace=True)
training_data['Embarked']=training_data['Embarked'].astype('category')

training_data['Sex']=training_data['Sex'].astype('category')
training_data['Embarked']=training_data['Embarked'].cat.codes

training_data['Sex']=training_data['Sex'].cat.codes

training_data.head()
training_data.shape
#dropping the columns which are not needed and do not impact the analysis

training_data.drop(columns=['Ticket','Fare','Cabin'],axis=1,inplace=True)
training_data[['Pclass','Survived']].groupby(by='Pclass').sum().sort_values(by='Survived',ascending=False)
training_data[['Sex','Survived']].groupby(by='Sex').sum().sort_values(by='Survived',ascending=False)
training_data[['Embarked','Survived']].groupby(by='Embarked').sum().sort_values(by='Survived',ascending=False)
training_data[['Parch','Survived']].groupby(by='Parch').sum().sort_values(by='Survived',ascending=False)
training_data['AgeBracket']=pd.cut(training_data['Age'],8)
training_data[['AgeBracket','Survived']].groupby(by='AgeBracket').sum().sort_values(by='Survived',ascending=False)
training_data.isnull().sum()
training_data['IsAlone']=((training_data['SibSp']+training_data['Parch']+1)>0).astype(int)
training_data['Age'].fillna(value=training_data['Age'].mode()[0],inplace=True)
training_data['Embarked'].fillna(value=training_data['Embarked'].mode()[0],inplace=True)
training_data.drop(columns=['SibSp','Parch','AgeBracket','Name','PassengerId'],axis=1,inplace=True)
training_data.head()
training_data['Age']=(training_data['Age'] - training_data['Age'].min()) / (training_data['Age'].max() - training_data['Age'].min())
x=training_data[['Pclass','Sex','Age','Embarked','IsAlone']]

y=training_data['Survived']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

logre=LogisticRegression()

logre.fit(x_train,y_train)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kfold=KFold(n_splits=100,shuffle=False,random_state=None)

cvs=cross_val_score(estimator=logre,X=x,y=y,cv=kfold)
np.mean(cvs)
logre.fit(x,y)
y_pred=logre.predict(x_test)
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=400,n_jobs=-1,max_leaf_nodes=16)

rf.fit(x_train,y_train)

rf_y_pred=rf.predict(x_test)
print(metrics.accuracy_score(y_test,rf_y_pred))
rf.fit(x,y)
from sklearn import metrics

import seaborn as sns

confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

sns.heatmap(pd.DataFrame(confusion_matrix),annot=True,cmap="YlGnBu",fmt='g')

print(metrics.classification_report(y_test,y_pred))
print('The Jacaard Similarity score is {0}'.format(metrics.jaccard_similarity_score(y_test,y_pred)))
print('The accuracy score is {0}'.format(metrics.accuracy_score(y_test,y_pred)))
test_data=pd.read_csv('../input/test.csv')

x_testing=test_data.copy()
x_testing['Age'].fillna(x_testing['Age'].mode()[0],inplace=True)

x_testing['Embarked'].fillna(x_testing['Embarked'].mode()[0],inplace=True)

x_testing['Embarked']=x_testing['Embarked'].astype('category')

x_testing['Sex']=x_testing['Sex'].astype('category')

x_testing['Embarked']=x_testing['Embarked'].cat.codes

x_testing['Sex']=x_testing['Sex'].cat.codes

x_testing.drop(columns=['Ticket','Fare','Cabin'],axis=1,inplace=True)

x_testing['IsAlone']=((x_testing['SibSp']+x_testing['Parch']+1)>0).astype(int)

x_testing.drop(columns=['SibSp','Parch','Name','PassengerId'],axis=1,inplace=True)

x_testing['Age']=(x_testing['Age'] - x_testing['Age'].min()) / (x_testing['Age'].max() - x_testing['Age'].min())

survived_pred=pd.DataFrame(logre.predict(x_testing),columns=['Survived'])
result=pd.DataFrame(data={'PassengerId':test_data['PassengerId'],'Survived':survived_pred['Survived']})
result.head()
final_pred=pd.DataFrame(rf.predict(x_testing),columns=['Survived'])

final_pred.head()
final_result=pd.DataFrame(data={'PassengerId':test_data['PassengerId'],'Survived':final_pred['Survived']})
final_result.to_csv("../working/finalresult.csv", index=False)
final_result.head(30)