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
#Подключим необходимые библиотеки

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('mode.chained_assignment', None)

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import LabelEncoder

import pylab as pl
#Присвоим к переменным имеющиеся датасеты

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
#Посмотрим что из себя представляет датасет train

train
train.pivot_table('PassengerId', 'Pclass', 'Survived', 'count').plot(kind='bar', stacked=True)
fig, axes = plt.subplots(ncols=2)

train.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0], title='SibSp')

train.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')
train
train.PassengerId[train.Cabin.notnull()].count()
train.PassengerId[train.Age.notnull()].count()
train.Age = train.Age.median()
train[train.Embarked.isnull()]
MaxPassEmbarked = train.groupby('Embarked').count()['PassengerId']

train.Embarked[train.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]
train.PassengerId[train.Fare.isnull()]
train = train.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

train
label = LabelEncoder()

dicts = {}



label.fit(train.Sex.drop_duplicates()) #задаем список значений для кодирования

dicts['Sex'] = list(label.classes_)

train.Sex = label.transform(train.Sex) #заменяем значения из списка кодами закодированных элементов 



label.fit(train.Embarked.drop_duplicates())

dicts['Embarked'] = list(label.classes_)

train.Embarked = label.transform(train.Embarked)
train
test.Age[test.Age.isnull()] = test.Age.mean()

test.Fare[test.Fare.isnull()] = test.Fare.median() #заполняем пустые значения средней ценой билета

MaxPassEmbarked = test.groupby('Embarked').count()['PassengerId']

test.Embarked[test.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

result = pd.DataFrame(test.PassengerId)

test = test.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)



label.fit(dicts['Sex'])

test.Sex = label.transform(test.Sex)



label.fit(dicts['Embarked'])

test.Embarked = label.transform(test.Embarked)
test
target = train.Survived

train = train.drop(['Survived'], axis=1) #из исходных данных убираем Id пассажира и флаг спасся он или нет

kfold = 5 #количество подвыборок для валидации

itog_val = {} #список для записи результатов кросс валидации разных алгоритмов
train
ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = train_test_split(train, target, test_size=0.25) 
model_rfc = RandomForestClassifier(n_estimators = 80, max_features='auto', criterion='entropy',max_depth=4) #в параметре передаем кол-во деревьев

model_knc = KNeighborsClassifier(n_neighbors = 18) #в параметре передаем кол-во соседей

model_lr = LogisticRegression(penalty='l2', tol=0.01) 

model_svc = svm.SVC() #по умолчанию kernek='rbf'
pl.clf()

plt.figure(figsize=(8,6))

#SVC

model_svc.probability = True

probas = model_svc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC', roc_auc))

#RandomForestClassifier

probas = model_rfc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RandonForest',roc_auc))

#KNeighborsClassifier

probas = model_knc.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KNeighborsClassifier',roc_auc))

#LogisticRegression

probas = model_lr.fit(ROCtrainTRN, ROCtrainTRG).predict_proba(ROCtestTRN)

fpr, tpr, thresholds = roc_curve(ROCtestTRG, probas[:, 1])

roc_auc  = auc(fpr, tpr)

pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LogisticRegression',roc_auc))

pl.plot([0, 1], [0, 1], 'k--')

pl.xlim([0.0, 1.0])

pl.ylim([0.0, 1.0])

pl.xlabel('False Positive Rate')

pl.ylabel('True Positive Rate')

pl.legend(loc=0, fontsize='small')

pl.show()
model_rfc.fit(train, target)

result.insert(1,'Survived', model_rfc.predict(test))

result.to_csv('akylson_predictions.csv', index=False)