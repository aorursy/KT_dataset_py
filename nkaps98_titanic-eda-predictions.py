# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# Any results you write to the current directory are saved as output.
import matplotlib.pyplot 

import seaborn as sns
dataset = pd.read_csv('../input/train.csv')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
Y_train = train.iloc[:,1].values

train.drop(columns=['Survived','Name','Ticket'],axis=1,inplace=True)

train.drop(columns=['Embarked','Cabin'],axis=1,inplace=True)

train = pd.get_dummies(train,columns=['Sex'],drop_first=True)
train.drop(columns=['PassengerId'],axis=1,inplace=True)
mean_value=train['Age'].mean()

train['Age']=train['Age'].fillna(mean_value)



mean_val=train['Fare'].mean()

train['Fare']=train['Fare'].fillna(mean_val)
X_train = train.iloc[:,:].values
sns.heatmap(dataset.corr())
dataset.corr()
sns.barplot(x='Sex',y='Survived',data=dataset)
sns.stripplot(x='Fare',y='Survived',data=dataset)
sns.swarmplot(x='Fare',y='Survived',data=dataset)
sns.boxplot(x='Age',y='Survived',data=dataset)
sns.boxplot(x='Pclass',y='Survived',data=dataset)
sns.violinplot(x='Sex',y='Survived',data=dataset)
sns.boxplot(x='Parch',y='SibSp',data=dataset)
sns.lmplot(x='Age',y='Survived',data=dataset)
test.drop(columns=['Name','Ticket'],axis=1,inplace=True)

test.drop(columns=['Embarked','Cabin'],axis=1,inplace=True)

test.drop(columns=['PassengerId'],axis=1,inplace=True)

test = pd.get_dummies(test,columns=['Sex'],drop_first=True)
mean_value=test['Age'].mean()

test['Age']=test['Age'].fillna(mean_value)



mean_val=test['Fare'].mean()

test['Fare']=test['Fare'].fillna(mean_val)
X_test = test.iloc[:,:].values
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier()

dectree.fit(X_train, Y_train)

y_pred = dectree.predict(X_test)
gender = pd.read_csv('../input/gender_submission.csv')

y_test = gender.iloc[:,1].values
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()

random_forest.fit(X_train, Y_train)

y_pred = random_forest.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
import xgboost as xgb

model=xgb.XGBClassifier(random_state=1,learning_rate=0.01)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

model.score(X_test,y_test)
df_test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_pred})
submission.head()
filename = 'Titanic Predictions.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)