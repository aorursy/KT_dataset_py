# importing basic libraries and dataset

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pandas_profiling

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



train_data= pd.read_csv('../input/titanic/train.csv')



test_data= pd.read_csv('../input/titanic/test.csv')

test_data['Survived']= np.nan

full_data= pd.concat([train_data,test_data])
full_data.profile_report()
# missingno is a python library used to visualiza missing data

import missingno as msno

msno.matrix(full_data);
print("Percentages of missing values: ")

full_data.isnull().mean().sort_values(ascending = False)
from statistics import mode

full_data["Embarked"] = full_data["Embarked"].fillna(mode(full_data["Embarked"]))
sns.heatmap(full_data.corr(),cmap='viridis');
full_data['Fare'] = full_data.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median()))

full_data['Age'] = full_data.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
full_data['Cabin'].isna().sum()/len(full_data)
full_data.drop('Cabin',axis=1,inplace=True)
full_data.info()
embarked = pd.get_dummies(full_data[['Embarked','Sex']],drop_first=True)

full_data = pd.concat([full_data,embarked],axis=1)
Name1 = full_data['Name'].apply(lambda x : x.split(',')[1])
full_data['Title'] = Name1.apply(lambda x : x.split('.')[0])
full_data['Title'].value_counts(normalize=True)*100
full_data['Title'] = full_data['Title'].replace([ ' Don', ' Rev', ' Dr', ' Mme',' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',' the Countess', ' Jonkheer', ' Dona'], 'Other')
full_data['Title'].unique()
embarked = pd.get_dummies(full_data['Title'],drop_first=True)

full_data = pd.concat([full_data,embarked],axis=1)
full_data.drop(['PassengerId','Name','Sex','Ticket','Title','Embarked'],axis=1,inplace=True)
full_data.info()
test = full_data[full_data['Survived'].isna()].drop(['Survived'], axis = 1)

train = full_data[full_data['Survived'].notna()]
train = train.astype(np.int64)

test = test.astype(np.int64)
train.shape,test.shape
sns.countplot(x='Survived',data=train_data,hue='Sex');
sns.countplot(x='Survived',data=train_data,hue='Pclass');
sns.distplot(train['Age'],kde=False,color='darkred',bins=30);
sns.countplot(x='SibSp',data=train);
sns.countplot(x='Parch',data=train);
train['Fare'].hist(color='green',bins=40,figsize=(12,6))

plt.xlabel('Fare');
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix



X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'], axis = 1), 

                                                    train['Survived'], test_size = 0.2, 

                                                    random_state = 2)
logisticRegression = LogisticRegression(max_iter = 10000)

logisticRegression.fit(X_train, y_train)

predictions = logisticRegression.predict(X_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score,cross_val_predict
kf = KFold(n_splits = 5)

score = cross_val_score(logisticRegression, train.drop('Survived', axis = 1),train['Survived'], cv = kf)

print(f"Accuracy after cross validation is {score.mean()*100}")
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(units=12,activation='tanh'))

model.add(Dense(units=100,activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(units=100,activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(units=100,activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')



early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)



model.fit(x=X_train.values, 

          y=y_train.values, 

          epochs=600,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop]

          )
model_loss = pd.DataFrame(model.history.history)

model_loss.plot();
dnn_predictions = model.predict_classes(X_test)

print(classification_report(y_test,dnn_predictions))

print(confusion_matrix(y_test,dnn_predictions))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(classification_report(y_test,rfc_pred))

print(confusion_matrix(y_test,rfc_pred))
param_grid = { 

    'criterion' : ['gini', 'entropy'],

    'n_estimators': [100, 300,500],

    'max_features': ['auto', 'log2'],

    'max_depth' : [3,5, 7,9]    

}



from sklearn.model_selection import GridSearchCV

randomForest_CV = GridSearchCV(estimator = rfc, param_grid = param_grid, cv = 5)

randomForest_CV.fit(X_train, y_train)

grid_pred = randomForest_CV.predict(X_test)

print(classification_report(y_test,grid_pred))

print(confusion_matrix(y_test,grid_pred))
randomForest_CV.best_params_
from xgboost import plot_importance,XGBClassifier
xgb = XGBClassifier().fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print(confusion_matrix(y_test, xgb_pred))

print(classification_report(y_test, xgb_pred))
print("Feature Importance")

plot_importance(xgb);
test['Survived'] = logisticRegression.predict(test)

test['PassengerId'] = test_data['PassengerId']

test[['PassengerId', 'Survived']].to_csv('lm_submission.csv', index = False)
test['Survived'] = model.predict_classes(test.iloc[:,:12])

test[['PassengerId', 'Survived']].to_csv('dnn_submission.csv', index = False)
test['Survived'] = rfc.predict(test.iloc[:,:12])

test[['PassengerId', 'Survived']].to_csv('rfc_submission.csv', index = False)
test['Survived'] = xgb.predict(test.iloc[:,:12])

test[['PassengerId', 'Survived']].to_csv('xgb_submission.csv', index = False)