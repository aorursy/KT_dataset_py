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
import pandas as pd



titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic_test = pd.read_csv('/kaggle/input/titanic/test.csv')

gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
titanic_train.head()
titanic_train.describe()
titanic_train.isnull().sum()
titanic_train.Name.value_counts()
titanic_train.Ticket.value_counts()
titanic_train.Cabin.value_counts()
new_train = titanic_train.drop(['Name','Ticket','Cabin'],axis=1)

new_train
new_train['Age']=new_train['Age'].fillna(new_train['Age'].mean())

new_train.isnull().sum()
new_train['family_size'] = new_train['SibSp']+new_train['Parch']

new_train.family_size.value_counts()
new_train['FareBin'] = pd.qcut(new_train['Fare'], 4)

new_train['AgeBin'] = pd.qcut(new_train['Age'], 4)
new_train = new_train.drop(['Fare','Age'],axis=1)

new_train = new_train.dropna(subset=['Embarked'])

new_train
categorical_features = new_train.drop(['PassengerId','Survived','SibSp','Parch','family_size'],axis=1)
categorical_features
a = categorical_features.columns.tolist()

encoded_features = pd.get_dummies(categorical_features, columns=a)

encoded_features
esc_dummy_trap = encoded_features.drop(['Pclass_3','Sex_male','Embarked_S','FareBin_(31.0, 512.329]','AgeBin_(35.0, 80.0]'],axis=1)

esc_dummy_trap
X = new_train.drop(['Pclass','Sex','Embarked','FareBin','AgeBin'],axis=1)

X= pd.concat([X,esc_dummy_trap],axis=1)

X
X_train = X.drop(['Survived','PassengerId'],axis=1)

X_train
Y_train = X['Survived']

Y_train.dtype
X_train.isnull().sum()
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, Y_train)
Y_train_pred = classifier.predict(X_train)
from sklearn.model_selection import cross_val_score

cross_val_score(classifier, X_train, Y_train, cv=3, scoring="accuracy")
corr_matrix = X.corr()

corr_matrix['Survived'].sort_values(ascending=False)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(Y_train,Y_train_pred)

cm
acc_scr = accuracy_score(Y_train,Y_train_pred)

acc_scr
report = classification_report(Y_train, Y_train_pred)

print(report)
titanic_test
for i in titanic_test['Parch']:

    titanic_test['Parch'] = titanic_test['Parch'].replace(9, 6)
new_test = titanic_test.drop(['Name','Ticket','Cabin'],axis=1)
new_test['Age']=new_test['Age'].fillna(new_test['Age'].mean())

new_test.isnull().sum()
new_test['Fare']=new_test['Fare'].fillna(new_test['Fare'].mean())

new_test.isnull().sum()
new_test['family_size'] = new_test['SibSp']+new_test['Parch']

new_test.family_size.value_counts()
new_test['FareBin'] = pd.qcut(new_test['Fare'], 4)

new_test['AgeBin'] = pd.qcut(new_test['Age'], 4)

new_test = new_test.drop(['Age','Fare'],axis=1)
titanic_test.Parch.value_counts()
categorical_features_test = new_test.drop(['PassengerId','SibSp','Parch','family_size'],axis=1)

categorical_features_test
a_test = categorical_features_test.columns.tolist()

encoded_features_test = pd.get_dummies(categorical_features_test, columns=a_test)

encoded_features_test
esc_dummy_trap_test = encoded_features_test.drop(['Pclass_3','Sex_male','Embarked_S','FareBin_(31.5, 512.329]','AgeBin_(35.75, 76.0]'],axis=1)
new_test
X_test = new_test.drop(['Pclass','Sex','Embarked','FareBin','AgeBin'],axis=1)

X_test = pd.concat([X_test,esc_dummy_trap_test],axis=1)
X_test = X_test.drop(['PassengerId'],axis=1)
X_test
X_train
titanic_test.Parch.value_counts()
test_pred = classifier.predict(X_test)
output = pd.DataFrame({'PassengerId': titanic_test.PassengerId, 'Survived': test_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")