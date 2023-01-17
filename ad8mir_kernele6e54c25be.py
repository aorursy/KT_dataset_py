import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dftrain = pd.read_csv('/kaggle/input/titanic/train.csv')

dftest = pd.read_csv('/kaggle/input/titanic/test.csv')
dftrain.head(1)
dftest.head(1)
def criarColmn(valor):

    if valor == 3:

      return 1

    else:

      return 0



dftrain['Pclass_third'] =  dftrain['Pclass'].map(criarColmn)

dftest['Pclass_third'] =  dftest['Pclass'].map(criarColmn)
def criarColmn(valor):

    if valor == 'female':

      return 1

    else:

      return 0



dftrain['Sex_bin'] =  dftrain['Sex'].map(criarColmn)

dftest['Sex_bin'] =  dftest['Sex'].map(criarColmn)
def criarColmn(valor):

    if valor == 'S':

      return 1

    else:

      return 0



dftrain['Embarked_S'] =  dftrain['Embarked'].map(criarColmn)

dftest['Embarked_S'] =  dftest['Embarked'].map(criarColmn)
def criarColmn(valor):

    if valor == 0:

      return 0

    else:

      return 1



dftrain['SibSp_bin'] =  dftrain['SibSp'].map(criarColmn)

dftest['SibSp_bin'] =  dftest['SibSp'].map(criarColmn)
def criarColmn(valor):

    if valor == 0:

      return 0

    else:

      return 1



dftrain['Parch_bin'] =  dftrain['SibSp'].map(criarColmn)

dftest['Parch_bin'] =  dftest['SibSp'].map(criarColmn)
dftrain['Age'].fillna((dftrain['Age'].mean()), inplace=True)

dftrain['Age'].isnull().sum()
dftest['Age'].fillna((dftest['Age'].mean()), inplace=True)
dftest['Age'].isnull().sum()
dftest['Fare'].fillna((dftest['Fare'].mean()), inplace=True)
dftrain['Cabin'].value_counts()
def criarColmn(valor):

    if valor != 'NaN':

      return 1

    else:

      return 0



dftrain['Cabin_1'] =  dftrain['Cabin'].map(criarColmn)

dftest['Cabin_1'] =  dftest['Cabin'].map(criarColmn)
y_train = dftrain[['Survived']]
x_train = dftrain[['Age','Parch','Fare','Pclass_third','Sex_bin','Embarked_S','SibSp_bin']]
x_test = dftest[['Age','Parch','Fare','Pclass_third', 'Sex_bin','Embarked_S','SibSp_bin']]
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier()
model = model.fit(x_train,y_train)
model_train_pred = model.predict(x_train)

predictions = [round(value) for value in model_train_pred]
accuracy = accuracy_score(y_train, model_train_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



accuracy = accuracy_score(y_train, model_train_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
submission_file4.shape
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_train, model_train_pred)

cnf_matrix
print("Accuracy:",metrics.accuracy_score(y_train, model_train_pred))

print("Precision:",metrics.precision_score(y_train, model_train_pred))

print("Recall:",metrics.recall_score(y_train, model_train_pred))
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

y_train_pred_prob = model.predict_proba(x_train)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_train,  model_train_pred)

auc = metrics.roc_auc_score(y_train,  model_train_pred)

plt.plot(fpr,tpr,label="AUC="+str(auc))

plt.legend(loc=4)

plt.show()
y_test_pred = model.predict(x_test)
y_test_pred.shape
submission_file6 = pd.Series(y_test_pred, index = dftest['PassengerId'], name = 'Survived')
submission_file6.shape
submission_file6.to_csv("sumission_file6.csv", header = True)
!head -n10 sumission_file6.csv