# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df1=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df1
df3=pd.read_csv("/kaggle/input/titanic/test.csv")

df3
df2=pd.read_csv("/kaggle/input/titanic/train.csv")

df2
del df2['Name']
df2
df2.pivot_table('PassengerId', 'Pclass', 'Survived','count').plot(kind='bar', stacked=True)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2)

df2.pivot_table('PassengerId', ['SibSp'], 'Survived', 'count').plot(ax=axes[0], title='SibSp')

df2.pivot_table('PassengerId', ['Parch'], 'Survived', 'count').plot(ax=axes[1], title='Parch')
df2.PassengerId[df2.Cabin.notnull()].count()
df2.PassengerId[df2.Age.notnull()].count()
df_Age = df2.Age.median()
df_Age
df2['Age'].fillna((df2['Age'].mean()), inplace=True)

df2
df2[df2.Embarked.isnull()]
MaxPassEmbarked = df2.groupby('Embarked').count()['PassengerId']

df2.Embarked[df2.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

df2.Embarked[df2.PassengerId==830]

df2.Embarked[df2.PassengerId==62]
MaxPassEmbarked=df2.groupby('Embarked').count()['PassengerId'].max()

print(MaxPassEmbarked)
df2.PassengerId[df2.Fare.isnull()]
df2 = df2.drop(['PassengerId','Ticket','Cabin'],axis=1)
df2
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

dicts = {}



label.fit(df2.Sex.drop_duplicates())

dicts['Sex'] = list(label.classes_)

df2.Sex = label.transform(df2.Sex)



label.fit(df2.Embarked.drop_duplicates())

dicts['Embarked'] = list(label.classes_)

df2.Embarked = label.transform(df2.Embarked)
df2
df3['Age'].fillna((df3['Age'].mean()), inplace=True)

df3.Fare[df3.Fare.isnull()] = df3.Fare.median() #fulfill NaN values with median

MaxPassEmbarked = df3.groupby('Embarked').count()['PassengerId']

df3.Embarked[df3.Embarked.isnull()] = MaxPassEmbarked[MaxPassEmbarked == MaxPassEmbarked.max()].index[0]

result = pd.DataFrame(df3.PassengerId)

df3 = df3.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)



label.fit(dicts['Sex'])

df3.Sex = label.transform(df3.Sex)



label.fit(dicts['Embarked'])

df3.Embarked = label.transform(df3.Embarked)
df3
from sklearn import model_selection

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

import pylab as pl
target = df2.Survived

train = df2.drop(['Survived'], axis=1) #drop surviving 

kfold = 5 #validation subsamples

result_val = {} #list for results of validation
train
ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = model_selection.train_test_split(train, target, test_size=0.25) 
model_rfc = RandomForestClassifier(n_estimators = 70) #number of trees

model_knc = KNeighborsClassifier(n_neighbors = 18) #number of neighbors

model_lr = LogisticRegression(penalty='l1', tol=0.01) 

model_svc = svm.SVC()
scores = model_selection.cross_val_score(model_rfc, train, target, cv = kfold)

result_val['RandomForestClassifier'] = scores.mean()

scores = model_selection.cross_val_score(model_knc, train, target, cv = kfold)

result_val['KNeighborsClassifier'] = scores.mean()

scores = model_selection.cross_val_score(model_lr, train, target, cv = kfold)

result_val['LogisticRegression'] = scores.mean()

scores = model_selection.cross_val_score(model_svc, train, target, cv = kfold)

result_val['SVC'] = scores.mean()
pd.DataFrame.from_dict(data = result_val, orient='index').plot(kind='bar', legend=False)
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

result.insert(1,'Survived', model_rfc.predict(df3))

result.to_csv('result.csv', index=False)
result