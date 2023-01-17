import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.head()
numer_of_rows = train.shape[0]

numer_of_rows
missing_vaules = train.isnull().sum()

missing_vaules
train_drop = train.dropna(subset = ['Embarked'])

#numer_of_rows = train_drop.shape[0]

#numer_of_rows
mean_age = train_drop[['Age']].mean()

mean_age

train_fill_age = train_drop.fillna({'Age': 30 })

train_fill_age.isnull().sum()
train_fill_cabin = train_fill_age.fillna(method = 'bfill', axis=0)

train_fill_cabin.isnull().sum()

#train_fill_cabin.head()
train_filled = train_fill_cabin.dropna(subset = ['Cabin'])

train_filled.isnull().sum()
train_filled.head()
train_drop_num = train_filled.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

train_drop_num.head()
train_drop_num['Cabin'].replace(to_replace=r'[0-9]', value='', regex=True)

train_reg = train_drop_num 

train_reg = train_reg.assign(Cabin=(train_drop_num['Cabin'].replace(to_replace=r'[0-9]', value='', regex=True)))

train_reg.head()
train_cat = train_reg

labels, uniqes = pd.factorize(train_reg['Sex'])

train_cat['Sex']=labels

#train_test_cat = train_test_cat.assign(Sex=pd.factorize(train_reg['Sex']))
labels1, uniqes1 = pd.factorize(train_reg['Cabin'])

train_cat['Cabin']=labels1
labels2, uniqes2 = pd.factorize(train_reg['Embarked'])

train_cat['Embarked']=labels2
train_cat.head()
train_clean = train_cat

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#scaler.fit(train_selected.drop('Survived',axis=1))

#scaled_features = scaler.transform(train_selected.drop('Survived',axis=1))

scaler.fit(train_clean.drop('Survived',axis=1))

scaled_features = scaler.transform(train_clean.drop('Survived',axis=1))

x_train_scaled = scaled_features
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

#x = train_clean.drop('Survived',axis=1)

y_train = train_clean['Survived']

rfc = RandomForestClassifier()

rfc = rfc.fit(x_train_scaled, y_train)

sfm = SelectFromModel(rfc, prefit=True)

selected_features = sfm.transform(x_train_scaled)

selected_features

sfm.get_support(indices=True)

for feature_list_index in sfm.get_support(indices=True):

   print(labels[feature_list_index])
x_train_selected = selected_features

x_train_scaled = scaled_features

y_train = train_clean['Survived']
from sklearn.svm import SVC 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, cross_val_predict

Classifiers = [SVC(gamma= 'scale'), KNeighborsClassifier(), DecisionTreeClassifier(), 

                ExtraTreesClassifier(n_estimators=100), LogisticRegression(solver = 'lbfgs'), 

               RandomForestClassifier(n_estimators=100)]
Model = []

Cross_val_score = []

Accuracy = []

for clf in Classifiers:

    predictions = cross_val_predict(clf, x_train_selected, y_train, cv=10)

    score = cross_val_score(clf, x_train_selected, y_train, scoring='accuracy', cv = 10)

    accuracy = score.mean() * 100

    Model.append(clf.__class__.__name__)

    Cross_val_score.append(score)

    Accuracy.append(accuracy)

    print('Score of '+clf.__class__.__name__ +' : '+ str(score)+ '\n'+

          'Accuracy of '+clf.__class__.__name__ +' : '+ str(accuracy))
Model = []

Cross_val_score = []

Accuracy = []

for clf in Classifiers:

    predictions = cross_val_predict(clf, x_train_scaled, y_train, cv=10)

    score = cross_val_score(clf, x_train_scaled, y_train, scoring='accuracy', cv = 10)

    accuracy = score.mean() * 100

    Model.append(clf.__class__.__name__)

    Cross_val_score.append(score)

    Accuracy.append(accuracy)

    print('Accuracy of '+clf.__class__.__name__ +' : '+ str(accuracy))

   # print('Score of '+clf.__class__.__name__ +' : '+ str(score)+ '\n'+

          #'Accuracy of '+clf.__class__.__name__ +' : '+ str(accuracy))
param_grid = {"gamma": ["auto", 0.01, 0.1, 0.5, 1, 2, 10],

              "C": [0.001, 0.01, 0.1, 1, 10],

             }

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

gs = GridSearchCV(SVC(), param_grid, cv=StratifiedKFold(n_splits=5))

gs_result = gs.fit(x_train_scaled, y_train)

print(gs_result.best_score_)

print(gs_result.best_estimator_)

print(gs_result.best_params_)
test.head()
test.isnull().sum()
test_drop = test.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

#test_drop.isnull().sum()
mean_age = test_drop[['Age']].mean()

mean_age

test_fill_age = test_drop.fillna({'Age': 30 })

test_fill_age.isnull().sum()
mean_fare = test_fill_age[['Fare']].mean()

mean_fare

test_fill_fare = test_fill_age.fillna({'Fare': 35.6271 })

test_fill_fare.isnull().sum()
test_fill_cabin = test_fill_fare.fillna(method = 'bfill', axis=0)

test_fill_cabin.isnull().sum()



#test_fill_cabin.head()
#test_fill_cabin['Cabin'].replace(to_replace=r'[0-9]', value='', regex=True)

test_reg = test_fill_cabin

test_reg = test_reg.assign(Cabin=(test_fill_cabin['Cabin'].replace(to_replace=r'[0-9]', value='', regex=True)))

test_reg.isnull().sum()
#test_reg.count(['Cabin'])

test_reg.loc[test_reg.Cabin == 'C', 'Cabin'].count()

test_reg_cabin = test_reg.fillna({'Cabin': 'C' })

test_reg_cabin.head()
test_clean = test_reg_cabin

labels, uniqes = pd.factorize(test_reg_cabin['Sex'])

test_clean['Sex']=labels

labels1, uniqes1 = pd.factorize(test_reg_cabin['Cabin'])

test_clean['Cabin']=labels1

labels2, uniqes2 = pd.factorize(test_reg['Embarked'])

test_clean['Embarked']=labels2

test_clean.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#scaler.fit(train_selected.drop('Survived',axis=1))

#scaled_features = scaler.transform(train_selected.drop('Survived',axis=1))

scaler.fit(test_clean)

test_scaled_features = scaler.transform(test_clean)

#x_train = scaled_features

#y_train = train_cat['Survived']
x_train_scaled = scaled_features

y_train = train_clean['Survived']
svc_model = SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)

svc_model.fit(x_train_scaled,y_train)

predictions = svc_model.predict(test_scaled_features)
predictions
result_dict = {'PassengerId' : test['PassengerId'],

          'Survived': predictions}

result = pd.DataFrame(result_dict)

result.head()
result.to_csv('submission_svc.csv', index=False)