# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import train & test Data
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train.head()
train.info()
test.info()
dataset=[train,test]
train['Cabin'].unique()
len(train['Ticket'].unique())
for data in dataset:
    data['lastname']=data['Name'].apply(lambda x: x.split(',')[0])
len(train['lastname'].unique())
count=0
lastnamerecorder=[]
for i in range(train.shape[0]):
    if( train['lastname'][i] not in lastnamerecorder):
        count=count+1+train['SibSp'][i]+train['Parch'][i]
        lastnamerecorder.append(train['lastname'][i])
print(count)
lastname_cabin={}
for i in range(train.shape[0]):
    if( train['lastname'][i] not in lastname_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):
        lastname_cabin[train['lastname'][i]]=[train['Cabin'][i]]
    elif(train['lastname'][i] in lastname_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):
        if(train['Cabin'][i] not in lastname_cabin[train['lastname'][i]] ):
            lastname_cabin[train['lastname'][i]].append(train['Cabin'][i])
len(lastname_cabin.keys())
class_cabin={}
for i in range(train.shape[0]):
    if( train['Pclass'][i] not in class_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):
        class_cabin[train['Pclass'][i]]=[train['Cabin'][i]]
    elif(train['Pclass'][i] in class_cabin.keys() and train['Cabin'][i]==train['Cabin'][i] ):
        if(train['Cabin'][i] not in class_cabin[train['Pclass'][i]] ):
            class_cabin[train['Pclass'][i]].append(train['Cabin'][i])
class_cabin
train.groupby(['Pclass']).count()
for data in dataset:
    data["cabin"]=data['Cabin'].apply(lambda x: x[0] if x==x else x)
train['cabin']
d=['X','A','B','C','D','E','F','G']

for data in dataset:
    for i in range(data.shape[0]):
        if(data['Pclass'][i]==1 and data["cabin"][i]!=data["cabin"][i]):
            data['cabin'][i]=d[np.random.choice(np.arange(1, 6), p=[0.3,0.3,0.2,0.15,0.05])]
        elif(data['Pclass'][i]==2 and data["cabin"][i]!=data["cabin"][i]):
            data['cabin'][i]=d[np.random.choice(np.arange(4, 7), p=[0.4,0.3,0.3])]
        elif(data['Pclass'][i]==3 and data["cabin"][i]!=data["cabin"][i]):
            data['cabin'][i]=d[np.random.choice(np.arange(5, 8), p=[0.3,0.4,0.3])]
train['cabin']
sns.countplot(x='cabin',hue='Survived' ,data=train)
sns.countplot(x='Pclass',hue='Survived' ,data=train)
for data in dataset:
    data['Age']=data['Age'].fillna(data['Age'].mean())
train
test=test.drop(["Cabin"], axis=1)
train=train.drop(["Cabin"], axis=1)
test.info()
for data in dataset:
    data['family']=data['Parch']+data['SibSp']
train.info()
columns=['PassengerId','Name','Ticket','lastname']
train=train.drop(columns,axis=1)
test1=test.drop(columns, axis=1)
train.info()

train.info()
train=pd.get_dummies(train)
test1=pd.get_dummies(test1)
train
X=train.drop(['Survived'],axis=1)

Y=train['Survived']
train.info()
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.3, random_state = 42)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
# Train the model on training data
rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)
predictions
from sklearn.metrics import accuracy_score
accuracy_score(test_labels, predictions)
test1=test1.fillna(test1['Fare'].mean())
test1.info()
test1['cabin_T']=0
predictions = rf.predict(test1)
test['Survived']=predictions
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('Titanic_submission.csv', index=False)

