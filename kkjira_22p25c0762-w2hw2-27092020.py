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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
train_data
train_data.info()
# check possible values for all string columns
print('Sex:',set(train_data['Sex']))
print('\nCabin:', set(train_data['Cabin']))
print('\nEmbarked:', set(train_data['Embarked']))
print('\nTicket:', set(train_data['Ticket']))
# fill na
tr = train_data.fillna({'Age':train_data['Age'].mean(), 'Cabin':'NA', 'Embarked':'NA'})
# extract features from columns: Name, Cabin, Ticket
import re

def name_prefix(str):
    x = re.search('^(?:.*,)?(?:\s*)?(.*?\.)', str)
    if x:
        return x[1]
    else:
        return ''

def all_number(str):
    return int(str.isnumeric())
    
tr['Name_len'] = tr['Name'].apply(len)
tr['Name_prefix'] = tr['Name'].apply(name_prefix)
tr['Cabin_prefix'] = tr['Cabin'].str[0]
tr['Ticket_isnumeric'] = tr['Ticket'].apply(all_number)
tr
# categorical columns
for c in ['Sex','Name_prefix','Ticket','Cabin','Cabin_prefix','Embarked']:
    tr[c] = tr[c].astype('category')
    tr[c+'_code'] = tr[c].cat.codes

tr.info()
import sklearn
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Name_len', 'Ticket_isnumeric', 'Sex_code', 'Name_prefix_code', 'Ticket_code','Cabin_code','Cabin_prefix_code','Embarked_code']
X = tr[features]
y = tr['Survived']
model = DecisionTreeRegressor()
model.fit(X, y)
importance = sorted(list(enumerate(model.feature_importances_)), key=lambda x:x[1], reverse=True)

print('Sorted Important Features ----------------------')
for i,v in importance:
    print(f'Feature: ({i}) {features[i]} :score={v:.5f}')
    
fig = plt.figure()
fig.patch.set_facecolor('xkcd:white')
plt.bar(list(zip(*importance))[0], list(zip(*importance))[1])

plt.show()
# Cabin_prefix_code is not related at all, so remove it from features

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Name_len', 'Ticket_isnumeric',
            'Sex_code', 'Name_prefix_code', 'Ticket_code','Cabin_code', 'Embarked_code']
X = tr[features]
y = tr['Survived']

models = []
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Neural Network (MLP)', MLPClassifier()))

for name, model in models:
    score = cross_validate(model, X, y, cv=5, scoring=['accuracy', 'recall', 'precision', 'f1'])
    print(f'Algorithm: {name} ------------------------------------------------')
    for i in range(5):
        print(f'Fold #{i}: Acc: {score["test_accuracy"][i]:.4f}', end='')        
        print(f' | Recall: {score["test_recall"][i]:.4f}', end='')
        print(f' | Precision: {score["test_precision"][i]:.4f}', end='')        
        print(f' | F1: {score["test_f1"][i]:.4f}')
    print(f'Average F1: {score["test_f1"].mean():.4f}')
# แสดง Confusion Matrix พร้อมคำนวณหาค่า Recall, Precision, F1
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Name_len', 'Ticket_isnumeric',
            'Sex_code', 'Name_prefix_code', 'Ticket_code','Cabin_code', 'Embarked_code']
X = tr[features]
y = tr['Survived']

models = []
models.append(('Decision Tree', DecisionTreeClassifier()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Neural Network (MLP)', MLPClassifier()))

for name, model  in models:
    print(f'Algorithm: {name}')
    y_pred = cross_val_predict(model, X, y, cv=5)
    conf = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = conf.ravel()

    print('Confusion matrix:')
    print(conf)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f'Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}\n')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.info()
# fill na
ts = test_data.fillna({'Age':test_data['Age'].mean(), 'Fare':test_data['Fare'].median(), 'Cabin':'NA'})

# extract features from columns: Name, Cabin, Ticket
import re

def name_prefix(str):
    x = re.search('^(?:.*,)?(?:\s*)?(.*?\.)', str)
    if x:
        return x[1]
    else:
        return ''

def all_number(str):
    return int(str.isnumeric())
    
ts['Name_len'] = ts['Name'].apply(len)
ts['Name_prefix'] = ts['Name'].apply(name_prefix)
ts['Ticket_isnumeric'] = ts['Ticket'].apply(all_number)
ts
# categorical test columns
for c in ['Sex','Name_prefix','Ticket','Cabin', 'Embarked']:
    ts[c] = ts[c].astype('category')
    ts[c+'_code'] = ts[c].cat.codes

ts.info()
# predict with MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict

features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Name_len', 'Ticket_isnumeric',
            'Sex_code', 'Name_prefix_code', 'Ticket_code','Cabin_code', 'Embarked_code']
X_train = tr[features]
y_train = tr['Survived']
X_test = ts[features]

model = MLPClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

submission = pd.DataFrame({'PassengerId':ts['PassengerId'], 'Survived':y_pred})
submission.to_csv('test_submission.csv', index=False)
print(submission)