import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

print(train_data.shape)
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

test_data.head()

print(test_data.shape)
train_data.dtypes
train_data.isnull().sum()/len(train_data)*100
test_data.isnull().sum()/len(train_data)*100
train_data = train_data.drop(['Name','PassengerId','Cabin'],axis=1)



# not removing passengerId for test_data because necessary for submission file

test_data = test_data.drop(['Name','Cabin'],axis=1)
print(train_data['Age'].median())

print(test_data['Age'].median())
train_data['Age'] = train_data['Age'].fillna(float(28.0))

test_data['Age'] = test_data['Age'].fillna(float(27.0))
print(test_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(float(14.4542))
print(train_data['Embarked'].value_counts())
train_data['Embarked'] = train_data['Embarked'].fillna('S')
train_data.isnull().sum()/len(train_data)*100
test_data.isnull().sum()/len(test_data)*100
plt.figure(figsize=(10,6))

sns.heatmap(train_data.corr().round(2),annot=True)
cat_features = [feat for feat in train_data.columns if train_data[feat].dtype==object]
cat_features
# train_data['Ticket'].unique()

# alpha_tickets = [var for var in train_data['Ticket'] if var[0].isdigit()==False]
Sex_dummies = pd.get_dummies(train_data['Sex'], drop_first=True)

Embarked_dummies = pd.get_dummies(train_data['Embarked'], drop_first=True)

Pclass_dummies = pd.get_dummies(train_data['Pclass'], prefix='Pclass',drop_first=True)



train_data = pd.concat([train_data, Sex_dummies], axis=1)

train_data = pd.concat([train_data, Embarked_dummies], axis=1)

train_data = pd.concat([train_data, Pclass_dummies], axis=1)
test_Sex_dummies= pd.get_dummies(test_data['Sex'], drop_first=True)

test_Embarked_dummies = pd.get_dummies(test_data['Embarked'], drop_first=True)

test_Pclass_dummies = pd.get_dummies(test_data['Pclass'], prefix='Pclass',drop_first=True)



test_data = pd.concat([test_data, test_Sex_dummies], axis=1)

test_data = pd.concat([test_data, test_Embarked_dummies], axis=1)

test_data = pd.concat([test_data, test_Pclass_dummies], axis=1)
train_data = train_data.drop(['Sex','Ticket','Embarked','Pclass'],axis=1)

test_data = test_data.drop(['Sex','Ticket','Embarked','Pclass'],axis=1)
train_data.head()
test_data.head()
sns.distplot(train_data.Age)
sns.distplot(train_data.Fare)
sns.distplot(test_data.Fare)
train_data = train_data[train_data.Fare != 0]

# test_data = test_data[test_data.Fare != 0] ### can't do this because all ids need a prediction.
test_data[test_data.Fare == 0]
test_data[(test_data.Pclass_2 == 0) & (test_data.Pclass_2 == 0)]['Fare'].median()
test_data.loc[266,'Fare'] = 13.9

test_data.loc[372,'Fare'] = 13.9
test_data[test_data.Fare == 0]
train_data['Fare'] = np.log(train_data['Fare'])

test_data['Fare'] = np.log(test_data['Fare'])
sns.distplot(train_data.Fare)
plt.figure(figsize=(10,6))

sns.heatmap(train_data.corr().round(2),annot=True)
X = train_data.drop('Survived',axis=1)

y = train_data['Survived']
X_submission = test_data
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=101)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import plot_roc_curve

from sklearn.model_selection import train_test_split
svc = SVC(random_state=42)

svc.fit(X_train, y_train)



svc_disp = plot_roc_curve(svc, X_test, y_test)

plt.show()
rfc = RandomForestClassifier(n_estimators=10, random_state=42)

rfc.fit(X_train, y_train)



ax = plt.gca()

rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)

svc_disp.plot(ax=ax, alpha=0.8)

plt.show()
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import plot_confusion_matrix
training_predictions = svc.predict(X_train)

print(classification_report(y_train, training_predictions))

print(confusion_matrix(y_train, training_predictions))

plot_confusion_matrix(svc, X_train, y_train,normalize='true')
test_predictions = svc.predict(X_test)

print(classification_report(y_test, test_predictions))

print(confusion_matrix(y_test, test_predictions))

plot_confusion_matrix(svc, X_test, y_test,normalize='true')
X_submission.columns
y_submission = svc.predict(X_submission[['Age', 'SibSp', 'Parch',

                                          'Fare', 'male', 'Q', 'S',

                                          'Pclass_2', 'Pclass_3']])
output = pd.DataFrame({'PassengerId': X_submission.PassengerId, 'Survived': y_submission})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
y_submission.shape