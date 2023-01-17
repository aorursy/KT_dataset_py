# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
gender_sub = pd.read_csv('../input/gender_submission.csv')
test = pd.read_csv('../input/test.csv')
#showing the sample of train data 
train.head()
# describe the train dataset
train.describe()

#checking for null values in data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Count of survived and those who don't
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
# Those who survived (male /female)
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# survived on basis of class
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
# column has so much null values
train=train.drop('Cabin',axis=1)
train.head()
sns.countplot(x='SibSp',data=train)
# Average age and passanger class
plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
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
train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)
train.head()
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
plt.figure(figsize=(16, 10))
# this graph is showing that there is no null value in dataset
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# droping the cabin

test = test.drop('Cabin',axis=1)
#here axis 1 specifies that we are searching for columns if it is 0 then rows.
test.head()
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test= pd.concat([test,sex,embark],axis=1)
test.head()
plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass',y='Age',data=test,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 28

        else:
            return 24

    else:
        return Age
test['Age']=test[['Age','Pclass']].apply(impute_age,axis=1)
test.head()
plt.figure(figsize=(16, 10))
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.figure(figsize=(16, 10))
sns.boxplot(x='Pclass',y='Fare',data=test,palette='winter')
plt.ylim(0,100)
def impute_fare(cols):
    Fare = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Fare):

        if Pclass == 1:
            return 60

        elif Pclass == 2:
            return 16

        else:
            return 10

    else:
        return Fare
test['Fare']=test[['Fare','Pclass']].apply(impute_fare,axis=1)
X_train=train.drop('Survived',axis=1)
X_train.head()
y_train=train['Survived']
y_train.head()
y_test=gender_sub['Survived']
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
X_test=test
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, predictions)
passid=np.array(list(range(892,1310)))
df = pd.DataFrame({'PassengerId':passid,'Survived':predictions})
df.to_csv('submission.csv',index=False)
