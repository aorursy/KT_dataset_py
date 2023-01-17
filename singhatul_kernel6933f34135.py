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
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train.head()
sns.heatmap(train.isnull(),cbar=False)
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Embarked',data=train,palette='rainbow')
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def compute_age(col):

    age = col[0]

    P_class = col[1]

    if pd.isnull(age):

        if P_class == 1:

            return 37

        elif P_class ==2:

            return 29

        else:

            return 24

    else:

        return age

    
train['Age'] = train[['Age','Pclass']].apply(compute_age,axis=1)
sns.heatmap(train.isnull(),cbar=False)
train.drop('Cabin',axis=1,inplace=True)
train.head()
print("Number of rows before dropping any row with missing value is: ",train.shape[0])

train.dropna(inplace=True)

print("Number of rows after dropping any row with missing value is: ",train.shape[0])
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)

train.head()
sns.heatmap(test.isnull(),cbar=False)
test['Age'] = test[['Age','Pclass']].apply(compute_age,axis=1)

test.drop('Cabin',axis=1,inplace=True)

test.head()
sex_test = pd.get_dummies(test['Sex'],drop_first=True)

embark_test= pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

test = pd.concat([test,sex_test,embark_test],axis=1)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

train.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.20, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,prediction))

print(classification_report(y_test,prediction))
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()

dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test,dt_pred))

print(classification_report(y_test,dt_pred))
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)
print(confusion_matrix(y_test,rf_pre))

print(classification_report(y_test,rf_pre))
from sklearn.svm import SVC
clf = SVC(gamma='scale')

clf.fit(X_train,y_train)
clf_pre=clf.predict(X_test)
print(confusion_matrix(y_test,clf_pre))

print(classification_report(y_test,clf_pre))
test_prediction = rf.predict(test)
test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])

new_test = pd.concat([test, test_pred], axis=1, join='inner')
new_test.head()
df= new_test[['PassengerId' ,'Survived']]

df.to_csv('predictions.csv' , index=False)