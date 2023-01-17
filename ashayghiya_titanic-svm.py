# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import re

import numpy as np

import matplotlib.pyplot as plt

import pickle as pk

import seaborn as sns

df = pd.read_csv(r'../input/train.csv')

df.head(n=20)
df['Salutation'] = df['Name'].str.extract(r'(,.*?\.)',expand=False).str.strip(', .')

avg_age = df.groupby(['Salutation'],as_index=False)['Age'].mean()

df = df.merge(avg_age,how='inner',on='Salutation')

df['Age'] = df['Age_x'].fillna(df['Age_y'])

del df['Age_x'], df['Age_y']
def age_band(x):

    if x <= 12:

        return 12

    elif x <= 18:

        return 18

    elif x <= 25:

        return 25

    elif x <= 30:

        return 30

    elif x <= 35:

        return 35

    elif x <= 40:

        return 40

    elif x <= 50:

        return 50

    elif x <= 60:

        return 60

    else:

        return 61



def Pclass (x):

    if x == 1:

        return 'Upper'

    elif x == 2:

        return 'Middle'

    else:

        return 'Lower'



df['AgeBand'] = df['Age'].apply(lambda x: age_band(x))

df['Class'] = df['Pclass'].apply(lambda x: Pclass(x))
X = pd.get_dummies(df[['Sex','Embarked','Age','Class','Fare']])

y = df['Survived']
X.head()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=5)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train.values)

y_train = y_train.values

X_test = sc.transform(X_test.values)
models = []

acc = []

precision = []

recall = []

f1 = []
#Logistic Regression

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 0)

lr.fit(X_train, y_train)
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)

models.append('Logistic Regression')

acc.append(accuracy_score(y_test, lr.predict(X_test)))

precision.append(precision_score(y_test, lr.predict(X_test)))

recall.append(recall_score(y_test, lr.predict(X_test)))

f1.append(f1_score(y_test, lr.predict(X_test)))
# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'entropy', 

                                    random_state = 0)

dt.fit(X_train, y_train)
models.append('Decision Tree')

acc.append(accuracy_score(y_test, dt.predict(X_test)))

precision.append(precision_score(y_test, dt.predict(X_test)))

recall.append(recall_score(y_test, dt.predict(X_test)))

f1.append(f1_score(y_test, dt.predict(X_test)))
# Fitting SVM to the Training set

from sklearn.svm import SVC

svc = SVC(kernel = 'rbf', random_state = 0, probability=True)

svc.fit(X_train, y_train)
models.append('SVM')

acc.append(accuracy_score(y_test, svc.predict(X_test)))

precision.append(precision_score(y_test, svc.predict(X_test)))

recall.append(recall_score(y_test, svc.predict(X_test)))

f1.append(f1_score(y_test, svc.predict(X_test)))
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 350, criterion = 'entropy', 

                                    random_state = 0)

rf.fit(X_train, y_train)
models.append('Random Forest')

acc.append(accuracy_score(y_test, rf.predict(X_test)))

precision.append(precision_score(y_test, rf.predict(X_test)))

recall.append(recall_score(y_test, rf.predict(X_test)))

f1.append(f1_score(y_test, rf.predict(X_test)))
rf.feature_importances_
# Fitting AdaBoost Classification to the Training set

from sklearn.ensemble import AdaBoostClassifier

adb = AdaBoostClassifier(base_estimator=dt, n_estimators=50, 

                         algorithm='SAMME.R', random_state=40)

adb.fit(X_train, y_train)
models.append('Adaboost')

acc.append(accuracy_score(y_test, adb.predict(X_test)))

precision.append(precision_score(y_test, adb.predict(X_test)))

recall.append(recall_score(y_test, adb.predict(X_test)))

f1.append(f1_score(y_test, adb.predict(X_test)))
# Fitting Voting Classifier Classification to the Training set

from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=[('Logistic Regression',lr),

                                   ('SVM',svc),

                                   ('Decision Tree',dt),

                                   ('Random Forest',rf),

                                   ('AdaBoost',adb)], 

                       voting='hard')

                       #flatten_transform=True)

vc.fit(X_train, y_train)
models.append('Average Ensemble')

acc.append(accuracy_score(y_test, vc.predict(X_test)))

precision.append(precision_score(y_test, vc.predict(X_test)))

recall.append(recall_score(y_test, vc.predict(X_test)))

f1.append(f1_score(y_test, vc.predict(X_test)))
# Fitting Voting Classifier Classification to the Training set

from sklearn.ensemble import VotingClassifier

vc2 = VotingClassifier(estimators=[('Logistic Regression',lr),

                                   ('SVM',svc),

                                   ('Decision Tree',dt),

                                   ('Random Forest',rf),

                                   ('AdaBoost',adb)],

                      voting='soft',

                      flatten_transform=True)

                      #weights=[1,5,2,4,3])

vc2.fit(X_train, y_train)
models.append('Average Ensemble Soft')

acc.append(accuracy_score(y_test, vc2.predict(X_test)))

precision.append(precision_score(y_test, vc2.predict(X_test)))

recall.append(recall_score(y_test, vc2.predict(X_test)))

f1.append(f1_score(y_test, vc2.predict(X_test)))
model_dict = {'Models': models,

             'Accuracies': acc,

             'Precision': precision,

             'Recall': recall,

             'f1-score': f1}

model_df = pd.DataFrame(model_dict).sort_values(by='Accuracies', ascending=False)

model_df
df1 = pd.read_csv(r'../input/test.csv')

df1['Salutation'] = df1['Name'].str.extract(r'(,.*?\.)',expand=False).str.strip(', .')

df1 = df1.merge(avg_age,how='left',on='Salutation')

df1['Age'] = df1['Age_x'].fillna(df1['Age_y'])

del df1['Age_x'], df1['Age_y']

df1['Class'] = df1['Pclass'].apply(lambda x: Pclass(x))

df1['Fare'] = df1['Fare'].fillna(df['Fare'].mean())
X1 = pd.get_dummies(df1[['Sex','Embarked','Age','Class','Fare']])

X1 = sc.transform(X1.values)
df1['Survived'] = svc.predict(X1)

sub = df1[['PassengerId','Survived']]

sub.to_csv(r'submission.csv',index=False)

sub