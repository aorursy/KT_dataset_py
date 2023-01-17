import numpy as np 





import pandas as pd 

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import style

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.info()
total = train_df.isnull().sum() 

total
train_df = train_df.drop(['PassengerId'],axis=1)

test_passenger_id = pd.DataFrame(test_df.PassengerId)

test_passenger_id.head()
test_df=test_df.drop(['PassengerId'],axis=1)
train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
train_df.Age.fillna(train_df.Age.median(),inplace=True)

test_df.Age.fillna(test_df.Age.median(),inplace=True)
data = [train_df, test_df] 

for dataset in data:

    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

    dataset["IsAlone"] = np.where(dataset["relatives"] > 0, 0,1)

train_df['IsAlone'].value_counts()  
for dataset in data:

    dataset.drop(['SibSp','Parch'],axis=1,inplace=True)
top_value = 'S'

data = [train_df,test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(top_value)
data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)
gender = {'male':0,'female':1}

data = [train_df, test_df]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(gender)
train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1)

ports = {'S':0,'C':1,'Q':77}

data = [train_df, test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)
train_df['Age'] = train_df['Age'].astype(int)

test_df['Age'] = test_df['Age'].astype(int)
X = train_df.drop("Survived", axis=1)

y = train_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123)
dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

accu_dt = dt.score(X_train,y_train)

accu_dt = round(accu_dt*100,2)

accu_dt
rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)

accu_rf = rf.score(X_train,y_train)

accu_rf = round(accu_rf*100,2)

accu_rf
y_final = rf.predict(test_df)

submission = pd.DataFrame({

    'PassengerId': test_passenger_id['PassengerId'],

    'Survived': y_final

})

submission.head()

submission.to_csv('titanic.csv', index=False)