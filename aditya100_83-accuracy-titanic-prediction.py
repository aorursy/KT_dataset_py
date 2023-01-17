import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.head()
train_data.shape
p = sns.countplot(data=train_data, x='Sex')
p = sns.countplot(data=train_data, x='Embarked')

_ = plt.title('C = Cherbourg, Q = Queenstown, S = Southampton')
p = sns.countplot(data=train_data, x='Survived')

_ = plt.title('0 = No, 1 = Yes')
p = sns.countplot(data=train_data, x='Pclass')

_ = plt.title('1 = 1st, 2 = 2nd, 3 = 3rd')
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
model = XGBClassifier()
train_data["Sex"] = train_data["Sex"].fillna("NA")

train_data["Embarked"] = train_data["Embarked"].fillna("C")

test_data["Sex"] = test_data["Sex"].fillna("NA")

test_data["Embarked"] = test_data["Embarked"].fillna("C")

train_data[['Pclass', 'Age', 'SibSp', 'Fare']] = train_data[['Pclass', 'Age', 'SibSp', 'Fare']].fillna(0)

test_data[['Pclass', 'Age', 'SibSp', 'Fare']] = test_data[['Pclass', 'Age', 'SibSp', 'Fare']].fillna(0)
genders = {'male': 0, 'female': 1, 'NA': 2}

embarks = {'C': 0, 'Q': 1, 'S': 2,}

train_data['Sex'] = train_data['Sex'].apply(lambda x: genders[x])

train_data['Embarked'] = train_data['Embarked'].apply(lambda x: embarks[x])

test_data['Sex'] = test_data['Sex'].apply(lambda x: genders[x])

test_data['Embarked'] = test_data['Embarked'].apply(lambda x: embarks[x])
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked', 'Fare']]

Y = train_data['Survived']
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()



X = sc.fit_transform(X)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)
model.fit(trainX, trainY)
predict = model.predict(testX)
from sklearn.metrics import accuracy_score
accuracy_score(predict, testY)
from sklearn.ensemble import RandomForestClassifier
model_f = RandomForestClassifier()
model_f.fit(trainX, trainY)
predict_f = model_f.predict(testX)
accuracy_score(predict_f, testY)
from sklearn.svm import SVC
model_svm = SVC(gamma='scale', decision_function_shape='ovo')
model_svm.fit(trainX, trainY)
predict_svm = model_svm.predict(testX)
accuracy_score(predict_f, testY)
from sklearn.neural_network import MLPClassifier
model_nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
model_nn.fit(trainX, trainY)
predict_nn = model_nn.predict(testX)
accuracy_score(predict_nn, testY)
test_df = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Embarked', 'Fare']]

test_df = sc.transform(test_df)

final_predictions = model_svm.predict(test_df)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": final_predictions

    })

submission.to_csv('submission.csv', index=False)