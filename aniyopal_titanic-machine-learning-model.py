# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
df_sur_train = df_train['Survived']
df_pas_test = df_test["PassengerId"]
df_sur_test = pd.read_csv('gender_submission.csv')['Survived']


df_train = df_train.drop(columns =['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch'])
df_test = df_test.drop(columns =['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch'])


df_train.Age = df_train.Age.fillna(df_train['Age'].mean())
df_train.Embarked = df_train.Embarked.fillna(method = 'bfill')

df_test.Age = df_test.Age.fillna(df_test['Age'].mean())
df_test.Fare = df_test.Fare.fillna(method = 'ffill')

df_train['Fare'] = df_train['Fare'].astype(int)
df_test['Fare'] = df_test['Fare'].astype(int)
 
df_train['Pclass'] = df_train['Pclass'].astype(str)
df_test['Pclass'] = df_test['Pclass'].astype(str)

ports = {"S": 0, "C": 1, "Q": 2}
data = [df_train, df_test]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)
    
genders = {"male": 0, "female": 1}
data = [df_train, df_test]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
    
X_train = df_train.drop("Survived", axis=1)
Y_train = df_sur_train
X_test  = df_test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X_train)
y_t =  sc.transform(X_test)

#Training the XGBoost model on
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(y_t)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(classifier, X, Y_train, cv=3)
confusion_matrix(Y_train, predictions)

submission = pd.DataFrame({"PassengerId": df_pas_test, "Survived": y_pred})

submission.to_csv('submission.csv', index=False)

