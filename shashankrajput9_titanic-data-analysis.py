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
import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
train =pd.read_csv("../input/titanic/train.csv")

test =pd.read_csv("../input/titanic/test.csv")



train.describe(include='all')
train.head(5)
train.info()
train.isnull().sum()
sns.barplot(x="Sex", y="Survived", data=train)
sns.barplot(x="Pclass", y="Survived", data=train)
sns.barplot(x="SibSp", y="Survived", data=train)
sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
train['CabinBool']=train["Cabin"].notnull()

test['CabinBool']=test["Cabin"].notnull()

sns.barplot(x="CabinBool", y="Survived", data=train)

plt.show()
train.info()
print(train.Embarked [train.Embarked == 'S'].count())

print(train.Embarked [train.Embarked == 'C'].count())

print(train.Embarked [train.Embarked == 'Q'].count())
train = train.fillna({"Embarked": "S"})



embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()


train['Sex'] = train['Sex'].map({"male": 0, "female": 1})

test['Sex'] = test['Sex'].map({"male": 0, "female": 1})



train.head()
train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
train.Age.fillna(value=train.Age.mean(), inplace=True)

train.Fare.fillna(value=train.Fare.mean(), inplace=True)



test.Age.fillna(value=test.Age.mean(), inplace=True)

test.Fare.fillna(value=test.Fare.mean(), inplace=True)
train['CabinBool'] = train['CabinBool'].map({True: 0, False: 1})

test['CabinBool'] = test['CabinBool'].map({True: 0, False: 1})
train.head()


bins = [ 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = [ 'Baby', 'Child', 'Teenager', 'Student', 'Young', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
train=train.drop(['Age'],axis =1)

test =test.drop(['Age'],axis=1)

train.head()
sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()

train.isnull().sum()
train.head()
from sklearn.model_selection import train_test_split



X = train.drop(['Survived', 'PassengerId'], axis=1)

y = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(X, y)

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



log_model = LogisticRegression()

log_model.fit(x_train, y_train)

y_pred = log_model.predict(x_val)

acc_log=accuracy_score(y_pred, y_val) * 100

print(acc_log)
from sklearn.svm import SVC

svm_model =SVC()

svm_model.fit(x_train,y_train)

y_pred =svm_model.predict(x_val)

acc_svc =accuracy_score(y_pred,y_val)*100

print(acc_svc)



from sklearn.tree import DecisionTreeClassifier



decisiontree_model =DecisionTreeClassifier()

decisiontree_model.fit(x_train,y_train)

y_pred =decisiontree_model.predict(x_val)



acc_decisiontree_model=accuracy_score(y_pred, y_val)*100

print(acc_decisiontree_model)
from sklearn.ensemble import RandomForestClassifier



randomforest_model = RandomForestClassifier()

randomforest_model.fit(x_train, y_train)

y_pred = randomforest_model.predict(x_val)

acc_randomforest =accuracy_score(y_pred, y_val) * 100

print(acc_randomforest)
from sklearn.neighbors import KNeighborsClassifier



knn_model = KNeighborsClassifier()

knn_model.fit(x_train, y_train)

y_pred = knn_model.predict(x_val)

acc_knn_model = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn_model)
from sklearn.linear_model import SGDClassifier



sgd_model = SGDClassifier()

sgd_model.fit(x_train, y_train)

y_pred = sgd_model.predict(x_val)

acc_sgd_model = accuracy_score(y_pred, y_val) * 100

print(acc_sgd_model)
from sklearn.ensemble import GradientBoostingClassifier



gbk_model = GradientBoostingClassifier()

gbk_model.fit(x_train, y_train)

y_pred = gbk_model.predict(x_val)

acc_gbk_model= accuracy_score(y_pred, y_val) * 100

print(acc_gbk_model)
compare =pd.DataFrame({

    'model':['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],

    'Score': [acc_svc, acc_knn_model, acc_log, 

              acc_randomforest, acc_decisiontree_model,

              acc_sgd_model, acc_gbk_model]

})
compare
ids = test['PassengerId']

predictions = gbk_model.predict(test.drop('PassengerId', axis=1))



output_file = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output_file.to_csv('submission.csv', index=False)