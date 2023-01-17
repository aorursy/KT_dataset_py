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
#importing visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#uploading data files
gs=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
tit_train=pd.read_csv('/kaggle/input/titanic/train.csv')
tit_test=pd.read_csv('/kaggle/input/titanic/test.csv')
# reading train dataset 
tit_train.head()
tit_train.info()
tit_train.describe()
sns.countplot(tit_train['Survived'])
plt.show()

## Gives insights about how many survived or not
sns.countplot(data=tit_train,x='Survived',hue='Sex')
plt.show()
## Shows female passengers survived more
sns.countplot(data=tit_train,x='Survived',hue='Pclass',palette='rainbow')
plt.show()

## Shows passengers of class 3 died the most and of class 1 survived the most
sns.countplot(data=tit_train,x='Survived',hue='Embarked',alpha=0.7)
plt.show()
sns.set_style('whitegrid')
sns.distplot(tit_train['Age'],kde=False,bins=50)
plt.show()

# Shows most of the passengers were youngsters (20-47)
plt.hist(tit_train['Fare'],bins=50)
plt.xlabel('Fare')
plt.show()

#Shows Fare distribution
sns.jointplot(y='Fare',x='Age',data=tit_train)
plt.show()
sns.countplot(tit_train['SibSp'])
plt.show()
sns.heatmap(tit_train.isnull(),cmap='summer')
plt.show()
# As seen Age , Cabin and Embarked contain null values..
# Lets fill them up !
tit_train.columns
# Lets try to fill Age column
sns.boxplot(y='Age',x='Pclass',data=tit_train)
# Getting avg values of age on the basis of Pclass is:
# Pclass 1: 37
# Pclass 2: 29
# Pclass 3:24
# Defining a function to fill NaN
def fill_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
# Now applying this fun to Age col 

tit_train['Age']=tit_train[['Age','Pclass']].apply(fill_age,axis=1)
# in test too

tit_test['Age']=tit_test[['Age','Pclass']].apply(fill_age,axis=1)
sns.heatmap(tit_train.isnull())
plt.show()

#no nulls in Age :D
sns.heatmap(tit_test.isnull())
plt.show()

# Fare contains null value
tit_test.head()
tit_train.head()
tit_test.head()
## Lets delete useless columns from both

tit_test.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
tit_train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
tit_test.columns
tit_train.columns
sns.heatmap(tit_test.isnull())
sns.heatmap(tit_train.isnull())
tit_test[tit_test['Fare'].isnull()]
def fill_fare(col):
    if(pd.isnull(col)):
        return 14.25
    else:
        return col
tit_test['Fare']=tit_test['Fare'].apply(fill_fare)
## Heatmap after cleaning null values
sns.heatmap(tit_test.isnull())
plt.show()
tit_train.info()

# 2 categorical cols exist
# Lets convert them into num
tit_train=pd.get_dummies(data=tit_train,columns=['Sex','Embarked'],drop_first=True)
tit_test=pd.get_dummies(data=tit_test,columns=['Sex','Embarked'],drop_first=True)
tit_train.dropna(inplace=True)
tit_train.head()
tit_test.head()
# Lets split our training set to test first
from sklearn.model_selection import train_test_split

X=tit_train.drop('Survived',axis=1)
y=tit_train['Survived']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
# Since its binary classification we can use Logistic Regression
# So lets import it
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=500)
rf.fit(x_train,y_train)
out=rf.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,out))
print('\n')
print(confusion_matrix(y_test,out))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,out))
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression(solver='liblinear')
lg.fit(x_train,y_train)
out_lg=lg.predict(x_test)
print(classification_report(y_test,out_lg))
print('\n')
print(confusion_matrix(y_test,out_lg))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,out_lg))
from sklearn.neighbors import KNeighborsClassifier
km=KNeighborsClassifier(n_neighbors=32)
km.fit(x_train,y_train)
out_km=km.predict(x_test)
print(classification_report(y_test,out_lg))
print('\n')
print(confusion_matrix(y_test,out_lg))
## Random forest gae us highest accuracy so let's apply that to our test data ;)
## Defining features and labels

predictions=rf.predict(tit_test)
predictions
submission=pd.DataFrame({
    'PassengerId':tit_test['PassengerId'],
    'Survived':predictions
})
submission
## DONE