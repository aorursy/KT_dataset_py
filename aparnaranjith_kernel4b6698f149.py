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
y_test_ex=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
y_test_ex.head()
train_df=pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.head(10)
train_df.info()
import seaborn as sns
sns.barplot(x='Survived',y='Pclass',hue='Sex',data=train_df)
sns.boxplot(x='Pclass',y='Age',hue='Sex',data=train_df)
sns.distplot(train_df['Age'])
sns.swarmplot(x='Sex',y='Age',hue='Survived',data=train_df)
sns.boxplot(x='Sex',y='Age',hue='Survived',data=train_df)
sns.boxplot(x='Sex',y='Parch',hue='Survived',data=train_df)
train_df['Age'].fillna(0,inplace=True)
train_df['Cabin'].fillna('no cabin',inplace=True)
train_df['Embarked'].fillna('no',inplace=True)
train_df.info()
X_test=pd.read_csv("/kaggle/input/titanic/test.csv")
X_test.head()

X_test['Age'].fillna(0,inplace=True)
X_test['Cabin'].fillna('nocabin',inplace=True)
X_test['Fare'].fillna(0,inplace=True)
X_test=X_test.drop(['Name','Sex','Ticket','Cabin','Embarked'],axis=1)
X_test.info()

X_test.head()
train_df.info()
X_train=train_df.drop(['Survived','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
X_train
X_train.info()
X_test.info()
y_train=train_df['Survived']


X_test.shape
#x=pd.get_dummies(X_train, columns=['Name','Sex','Ticket','Cabin','Embarked'],drop_first=True)

#x_test=pd.get_dummies(X_test, columns=['Name','Sex','Ticket','Cabin','Embarked'],drop_first=True)

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=7)
X_train
X_test
from sklearn import preprocessing
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
from sklearn import svm
#logisticRegr = LogisticRegression()
#logisticRegr.fit(X_train_scaled, y_train)
model=svm.SVC(kernel='rbf',C=80)
model.fit(X_train_scaled,y_train)
model.score(X_train_scaled,y_train)
prediction = model.predict(X_test_scaled)
prediction

result=pd.DataFrame(index=(j for j in range(0,418)),columns=['PassengerId','Survived'])
for i in range(0,418):
    result.iloc[i,0]=X_test.iloc[i,0]
    result.iloc[i,1]=prediction[i]
result.set_index('PassengerId')
result.to_csv('result_svm.csv')