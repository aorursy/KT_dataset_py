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


os.getcwd()
trainset = pd.read_csv('../input/titanic/train.csv')
testset = pd.read_csv('../input/titanic/test.csv')
gender_sub = pd.read_csv('../input/titanic/gender_submission.csv')
trainset['Sex_f'] = pd.factorize(trainset['Sex'])[0]
trainset['Embarked_f'] = pd.factorize(trainset['Embarked'])[0]

testset['Sex_f'] = pd.factorize(testset['Sex'])[0]
testset['Embarked_f'] = pd.factorize(testset['Embarked'])[0]
trainset2 = trainset[['Survived','Pclass','SibSp','Parch','Sex_f','Embarked_f']]
testset2 = testset[['Pclass','SibSp','Parch','Sex_f','Embarked_f']]
x = trainset2.iloc[:,1:]
y = trainset2.iloc[:,0]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression()

model.fit(x,y)

#x.isnull().any()
#x.isnull().sum().sum()

print(model.predict_proba(x))
print(model.score(x, y))
print(confusion_matrix(y, model.predict(x)))

results = pd.Series(model.predict(testset2))
#print(type(results))
print(gender_sub)
finalpred = pd.DataFrame({'PassengerId':pd.Series(testset.iloc[:,0]),'Survived':results})
print(finalpred)
finalpred.to_csv('LogisticFinal.csv')
from sklearn import svm

SVM = svm.LinearSVC()
SVM.fit(x, y)
results = pd.Series(SVM.predict(testset2))

#print(SVM.predict_proba(x))
print(SVM.score(x, y))
print(confusion_matrix(y, SVM.predict(x)))
from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(x, y)
results = pd.Series(RF.predict(testset2))

print(RF.score(x, y))
print(confusion_matrix(y, RF.predict(x)))
from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(x, y)
results = pd.Series(NN.predict(testset2))

print(NN.score(x, y))
print(confusion_matrix(y, NN.predict(x)))