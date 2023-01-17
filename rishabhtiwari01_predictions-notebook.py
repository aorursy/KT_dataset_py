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
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
output_file = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
output_file.head()
train.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
plt.figure(figsize=(10,10))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='summer')
def cabin_alloted(cabin):

    if pd.isnull(cabin):

        return 0

    else:

        return 1
train['Cabin Alloted'] = train['Cabin'].apply(cabin_alloted)
train.head()
first =  train[train['Pclass'] ==1 ]['Age'].mean()

second = train[train['Pclass'] ==2 ]['Age'].mean() 

third = train[train['Pclass'] ==3 ]['Age'].mean() 
def age_filler(data):

    pclass = data[0]

    age = data[1]

    

    if pd.isnull(age):

        if pclass == 1:

            return first

        elif pclass == 2:

            return second

        elif pclass == 3:

            return third

    else:

        return age
train['Age'] = train[['Pclass','Age']].apply(age_filler,axis=1)
plt.figure(figsize=(10,10))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='summer')
train.head()
sns.countplot(train['Embarked'])
train['Embarked'] = train['Embarked'].apply(lambda x: 'S' if pd.isnull(x) else x)
plt.figure(figsize=(10,10))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='summer')
train = train.merge(right=pd.get_dummies(train['Embarked']),right_index=True, left_index=True)
train.head()
train.columns
train.drop(['PassengerId','Pclass','Name','Sex','Ticket','Cabin','Embarked','female'],axis=1,inplace=True)
train.head()
##train.drop('female',axis=1,inplace=True)
train.head()
train.columns
X = train[['Age','SibSp','Parch','Fare','Cabin Alloted',1,2,3,'male','C','Q','S']]
y = train['Survived']
y
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(X,y)
model.coef_
output = pd.DataFrame(data=model.coef_,columns=X.columns,index=['coefs'])
output
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head(2)
plt.figure(figsize=(10,10))

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='summer')
test[pd.isnull(test['Fare'])]
test.loc[152,'Fare'] = test[test['Pclass']==3]['Fare'].mean()
test.loc[152,'Fare']
test.head()
test = test.merge(right=pd.get_dummies(test['Embarked']),right_index=True, left_index=True)
test['Cabin Alloted'] = test['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)

test['Age'] = test[['Pclass','Age']].apply(age_filler,axis=1)
test.head()
test.drop(['Pclass','Name','Sex','Ticket','Embarked','female','Cabin'],inplace=True,axis=1)

test.head()

test.columns
X_test = test[['Age','SibSp','Parch','Fare','Cabin Alloted',1,2,3,'male','C','Q','S']]
X_test.head()
test['survived']= model.predict(X_test)
predictions = test[['PassengerId','survived']]
output_file.head()
predictions.columns = ['PassengerId','Survived']
predictions
sns.countplot(predictions['Survived'])
test
sns.jointplot(x='Age',y='survived',data=test)
predictions.head()
import os
os.chdir('/kaggle/working')
#pwd
predictions.to_csv('predictions.csv',index=False)
#output_file
#predictions.head()