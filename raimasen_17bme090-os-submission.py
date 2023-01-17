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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.drop('Cabin',axis = 1,inplace = True)

test_data.drop('Cabin',axis = 1,inplace = True)
sns.heatmap(train_data.isnull(),yticklabels=False, cbar=False)
sns.heatmap(train_data.corr(), annot=True)
'''def fill_age_train(cols):

    Age = cols[5]

    if pd.isnull(Age):

        return Age.mean()

    else:

        return Age'''
'''def fill_age_test(cols):

    Age = cols[4]

    if pd.isnull(Age):

        return Age.mean()

    else:

        return Age'''
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
sns.heatmap(test_data.isnull(),yticklabels=False, cbar=False)
sns.boxplot(x = 'Pclass', y = 'Age', data = train_data)
new_train = train_data.dropna()
new_test = test_data.fillna(test_data['Fare'].mean())
sns.heatmap(new_train.isnull(),yticklabels=False, cbar=False)
new_train.shape
new_test.shape
new_train.head()
sex = pd.get_dummies(new_train['Sex'], drop_first = True)

pclass = pd.get_dummies(new_train['Pclass'], drop_first = True)

embark = pd.get_dummies(new_train['Embarked'], drop_first = True)
new_train = pd.concat([new_train,sex,pclass,embark],axis = 1)

new_train.drop(['Pclass','Name','Sex','Ticket','Embarked'],axis = 1,inplace = True)
new_train.head()
sex_test = pd.get_dummies(new_test['Sex'], drop_first = True)

pclass_test = pd.get_dummies(new_test['Pclass'], drop_first = True)

embark_test = pd.get_dummies(new_test['Embarked'], drop_first = True)
new_test = pd.concat([new_test,sex_test,pclass_test,embark_test],axis = 1)

new_test.drop(['Pclass','Name','Sex','Ticket','Embarked'],axis = 1,inplace = True)
new_test.head()
new_train.shape
new_test.shape
from sklearn.model_selection import train_test_split



X = new_train.drop(['Survived'],axis = 1).values

y = new_train['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.svm import SVC
model = SVC()

model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
param_grid = {'C': [0.1,1, 10, 100, 1000,10000,100000], 'gamma': [1,0.1,0.01,0.001,0.0001,0.00001,0.000001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
grid.fit(X_train,y_train)
grid.best_params_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
new_test.head()
grid_predictions_new = grid.predict(new_test)
#tests = model.predict()
#tests
output = pd.DataFrame({'PassengerId': new_test.PassengerId,'Survived': grid_predictions_new})
output
output.to_csv('my_new.csv', index=False)
'''from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")'''