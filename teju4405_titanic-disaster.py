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
train=pd.read_csv('/kaggle/input/titanic/train.csv').drop(columns={'Ticket','Fare','Cabin'}).dropna()

train.head()
train['Sex']=pd.get_dummies(train['Sex'])

train.head()
train['Embarked']=train['Embarked'].replace({'S':0,'C':1,'Q':2},regex=True)

train.head()
import seaborn as sns

from matplotlib import pyplot as plt

sns.barplot(data=train,x='Embarked',y='Survived',hue='Sex')

sns.barplot(data=train,x='Embarked',y='Survived',hue='Pclass')


sns.heatmap(test.corr(),vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200))
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

x=train[['Sex','Embarked','Parch','SibSp','Pclass','Age']]

y=train[['Survived']]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

rf=RandomForestClassifier(random_state=0,max_features=5).fit(x_train,y_train)

ran=rf.predict(x_test)
print('Accuracy Score for Random Forest',accuracy_score(y_test,ran))

print('Precision Score {:.2f}'.format(precision_score(y_test,ran)))

print('F1 Score {:.2f}'.format(f1_score(y_test,ran)))

print('Recall Score {:.2f}'.format(recall_score(y_test,ran)))
test=pd.read_csv('/kaggle/input/titanic/test.csv').drop(columns={'Ticket','Fare','Cabin'})

test.head()
test['Embarked']=test['Embarked'].replace({'S':0,'C':1,'Q':2},regex=True)

test.head()
test['Sex']=pd.get_dummies(test['Sex'])

test.head()
test.fillna(0,inplace=True)

test1=test[['Sex','Embarked','Parch','SibSp','Pclass','Age']]
tet=test[['PassengerId','Name']]

pred1=rf.predict(test1)

x=pd.DataFrame(pred1,columns=['Survived'])



mer=pd.merge(tet,x,how='outer',left_index=True,right_index=True).rename(columns={0:'Survived'})

mer.head()
submission=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

submission.to_csv('submission.csv', index=False)
submission.head()