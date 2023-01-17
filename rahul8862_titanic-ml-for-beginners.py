# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')
train.head()
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
plt.figure(figsize=(10,5))

sns.countplot(x='Survived',hue='SibSp',data=train)
sns.countplot(x='Survived',hue='Parch',data=train)
sns.countplot(x='Survived',hue='Embarked',data=train)
features= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

x = train[features]

y = train['Survived']
x.isnull().sum()
sns.heatmap(x.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=x)
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
x['Age'] = x[['Age','Pclass']].apply(impute_age,axis=1)
x.isnull().sum()
x['Embarked'].value_counts()
x['Embarked']= x['Embarked'].fillna(x['Embarked'].value_counts().index[0])
sex = pd.get_dummies(x['Sex'],drop_first=True)

embarked = pd.get_dummies(x['Embarked'],drop_first=True)
x.drop(['Sex','Embarked'],axis=1,inplace=True)
x.head()
x = pd.concat([x,sex,embarked],axis=1)
x.head()
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(x, y)
test=pd.read_csv('../input/titanic/test.csv')
features_test= [ 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

x_test = test[features]
x_test.isnull().sum()
x_test['Age'] = x_test[['Age','Pclass']].apply(impute_age,axis=1)
x_test['Fare'].fillna(35.6271,inplace=True)
x_test.isnull().sum()
sex_test = pd.get_dummies(x_test['Sex'],drop_first=True)

embarked_test = pd.get_dummies(x_test['Embarked'],drop_first=True)
x_test.drop(['Sex','Embarked'],axis=1,inplace=True)
test= pd.concat([x_test,sex_test,embarked_test],axis=1)
test.head()
prediction=classifier.predict(test)
lll=pd.read_csv('../input/titanic/gender_submission.csv')
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
accuracy_score(lll['Survived'],prediction)
confusion_matrix(lll['Survived'],prediction)
submission=pd.DataFrame({'PassengerId':lll['PassengerId'],'Survived':prediction})
submission.to_csv('submission3.csv',index=False)