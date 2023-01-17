# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
train.set_index('PassengerId')
train.isnull() # Value which shows True meaning our data is misi=sing or vive-versa
p=sns.heatmap(train.isnull(),cmap='Blues')

p.set_title('Missing data')
Choose_best_variable=train.corr()
sns.heatmap(Choose_best_variable,annot=True,cmap='coolwarm')
sns.boxplot(x=train['Fare'],y=train['Age']) # As it coorelates most but it didn't seems to be inconsistent
sns.boxplot(x=train['SibSp'],y=train['Age']) # Well the data seems to be much more hypothetical
sns.boxplot(x=train['Pclass'],y=train['Age']) 
sns.countplot(x=train['Sex'])
sns.countplot(x=train['Survived'],hue=train['Sex'])
sns.countplot(x=train['Survived'],hue=train['Pclass'])
sns.countplot(x=train['Pclass'])

# Plot showing how many people belonging to different class
plt.figure(figsize=(12,4))

sns.kdeplot(train['Fare'])
sns.countplot(x=train['SibSp'])
sns.countplot(x=train['SibSp'],hue=train['Survived'])
data=pd.concat([train,test])

data
sns.heatmap(data.isnull(),cmap='Blues')
def fill(x):

    age=x[0]

    clas=x[1]

    if pd.isnull(age):

        if clas==1:

            return round(data[data['Pclass']==1]['Age'].mean())

        elif clas==2:

            return round(data[data['Pclass']==2]['Age'].mean())

        else:

            return round(data[data['Pclass']==3]['Age'].mean())

    else:

        return age

data['Age']=data[['Age','Pclass']].apply(fill,axis=1)

data.head()
sns.heatmap(data.isnull(),cmap='Blues')
a=pd.get_dummies(data['Sex'],drop_first=True)

a # Converting it into 0 & 1 form to interpret easily
b=pd.get_dummies(data['Embarked'],drop_first=True)

b
data=pd.concat([data,a,b],axis=1)

data.head(1)
data.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1,inplace=True)

data.head()
# Reconverting the data

train=data[:len(train)]

test=data[:len(test)]

sns.heatmap(train.isnull(),cmap='Blues')
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix
y=train['Survived']

x=train.drop('Survived',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=101,test_size=0.3)
cross_val_score(RandomForestClassifier(),x,y).mean()
cross_val_score(LogisticRegression(),x,y).mean()
trainmodel=LogisticRegression()
trainmodel.fit(x_train,y_train)
data_to_test=test.drop('Survived',axis=1)
result=trainmodel.predict(data_to_test)
result
test_again=pd.read_csv('/kaggle/input/titanic/test.csv')
work=pd.DataFrame({'PassengerId': test_again['PassengerId'], 'Survived':result})

final_work=work.round(0).astype(int)
final_work.to_csv('Result.csv',index = False)

final_work
test
sns.heatmap(test.isnull(),cmap='Blues')