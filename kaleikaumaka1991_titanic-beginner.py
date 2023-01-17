#Import necessary libraries

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier



%matplotlib inline



#Read CSVs and make backup

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

train_df_copy = train_df.copy()



test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_df_copy = test_df.copy()



sub_df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

sub_df.copy = sub_df.copy()
#Initial data glimpse

train_df.info()
#Initial dat glimpse

train_df.head()
#Initial data glimpse

train_df.describe()
#Initial data visualization

sns.pairplot(train_df, hue='Survived')
#Manipulate Age data

train_df['Age'].fillna(inplace=True,value=train_df['Age'].mean())



#Manipulate categorical data

train_df['Sex'] = train_df['Sex'].astype('category')



#Combine columns

train_df['Fsize'] = train_df['SibSp'] + train_df['Parch']



#Delete columns

train_df.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'],inplace=True)



#Prepare data

train_df = pd.get_dummies(train_df,columns=['Sex'])

train_df.drop(columns=['Sex_female'],inplace=True)



#Split Data

x = train_df.loc[:,train_df.columns!='Survived']

y = train_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#Same for Test Data

test_df['Age'].fillna(inplace=True,value=test_df['Age'].mean())

test_df['Sex'] = test_df['Sex'].astype('category')

test_df['Fsize'] = test_df['SibSp'] + test_df['Parch']

test_df.drop(columns=['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'],inplace=True)

test_df = pd.get_dummies(test_df,columns=['Sex'])

test_df.drop(columns=['Sex_female'],inplace=True)

test_df['Fare'].fillna(inplace=True, value=x_test['Age'].mean())

#x_test = test_df.loc[:,test_df.columns]
#Train Model

r_model = LogisticRegression(solver='liblinear')

r_model.fit(x_train,y_train)



#Use model to make predictions

r_pred = r_model.predict(x_test)

pred_prob = r_model.predict_proba(x_test)



#Model Accuracy

r_acc = r_model.score(x_test,y_test)
#Visualize model performance

confusion_matrix(y_test, r_pred)
#Show accuracy with other model performance data

print(classification_report(y_test,r_pred))
#Need more research on this function. I believe it measures how model performs in real world.

from sklearn.model_selection import cross_val_score



print(cross_val_score(r_model, x_train, y_train, cv=3))
from sklearn.ensemble import RandomForestRegressor



rf_model = RandomForestRegressor(n_estimators=100, random_state=1)

rf_model.fit(X_train,y_train)

rf_pred = rf_model.predict(X_test)



#Show accuracy with other performance data

rf_acc = rf_model.score(X_test,y_test)
#Visualize performance - I want to find a better visual for this!

#confusion_matrix(y_test, rf_pred)
from sklearn.svm import SVC



svm_model = SVC(gamma = 'auto')

svm_model.fit(x_train,y_train)

svm_pred = svm_model.predict(x_test)

svm_acc = svm_model.score(x_test,y_test)
#Visualize model performance

confusion_matrix(y_test, svm_pred)

print(classification_report(y_test,svm_pred))
from sklearn.naive_bayes import GaussianNB



gnb_model = GaussianNB()

gnb_model.fit(x_train, y_train)

gnb_pred = gnb_model.predict(x_test)

gnb_acc = gnb_model.score(x_test,y_test)
print(classification_report(y_test,gnb_pred))
from sklearn.linear_model import SGDClassifier



sgd_model = SGDClassifier()

sgd_model.fit(x_train,y_train)

sgd_pred = sgd_model.predict(x_test)

sgd_acc = sgd_model.score(x_test,y_test)
models = pd.DataFrame({

    'Model': ['Logistical Regression', 'Random Forest', 'Support Vector Machine', 

              'Naive Bayes', 'Stochastic Gradient Descent'],

    'Score': [r_acc, rf_acc, svm_acc, 

              gnb_acc, sgd_acc]})

models.sort_values(by='Score', ascending=False)
#submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": Y_pred})

# submission.to_csv('../output/submission.csv', index=False)