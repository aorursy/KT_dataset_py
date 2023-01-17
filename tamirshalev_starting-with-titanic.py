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
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test= pd.read_csv("/kaggle/input/titanic/test.csv")

train.columns
sns.barplot(x=train['Pclass'], y=train['Survived'])
sns.barplot(x=train['Sex'], y=train['Survived'])
train_family = pd.Series(train['SibSp'] + train['Parch'], name='Family')

test_family = pd.Series(test['SibSp'] + test['Parch'], name='Family')

train['Family']=train_family

test['Family']=test_family

sns.barplot(x=train['Family'], y=train['Survived'])
sns.heatmap(train.corr(), annot=True)
# See Age distribution

sns.kdeplot(data=train['Age'], shade=True)
# Infant: 0-3 |  Child: 3-13 | 'Teenager': 13-28 and so on..



train['Age'] = train['Age'].fillna(train.Age.median())

test['Age'] = test['Age'].fillna(test.Age.median())



group_ranges = [0,3,13,18,30,64,np.inf]

age_groups = ['Infant', 'Child','Teenager', 'Mature', 'Adult', 'Senior']



train['AgeGroup'] = pd.cut(train['Age'], bins=group_ranges, labels = age_groups)

test['AgeGroup'] = pd.cut(test['Age'], bins=group_ranges, labels = age_groups)



sns.barplot(x=train['AgeGroup'], y=train['Survived'])
train = train.drop(['Cabin'], axis=1)

test = test.drop(['Cabin'], axis=1)
train.info()
#train['Embarked'].value_counts()

train = pd.get_dummies(data=train, columns=['Embarked'], drop_first=True)

test = pd.get_dummies(data=test, columns=['Embarked'], drop_first=True)

# By checking test['Embarked'].value_counts() we will get that 'S' is also the most frequent.
train.info()
test.info()
# Fill missing value for Fare. Only 1 value missing so just use the median.

fare_med = test.Fare.median()

test.Fare.fillna(fare_med, inplace=True)

test.info()
train.head()

sex_map = {'male':1, 'female':2}

age_group_map ={'Infant':1, 'Child':2, 'Teenager':3, 'Mature':4, 'Adult':5, 'Senior':6}



train['Sex'] = train['Sex'].map(sex_map)

train['AgeGroup'] = train['AgeGroup'].map(age_group_map).astype('int64')



test['Sex'] = test['Sex'].map(sex_map)

test['AgeGroup'] = test['AgeGroup'].map(age_group_map).astype('int64')



train.info()
sns.distplot(a=train['Fare'], kde=False)
rank_ranges = [0,8, 15, 30, np.inf]

fare_ranks = [1,2,3,4]



#train['FareRank'] = pd.cut(train['Fare'], bins=rank_ranges, labels = fare_ranks)

#test['FareRank'] = pd.cut(test['Age'], bins=rank_ranges, labels = fare_ranks)



train = train.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Age'], axis=1)

test = test.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'Age'], axis=1)



train['Fare'] = (train['Fare'] - train['Fare'].mean()) / train['Fare'].max()

test['Fare'] = (test['Fare'] - test['Fare'].mean()) / test['Fare'].max()





train.head()
y = train['Survived']

X = train.drop(['Survived'], axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.linear_model import Perceptron

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)

preds = log_reg.predict(X_valid)

lr_acc = accuracy_score(preds, y_valid)

print(lr_acc)
gauss = GaussianNB()

gauss.fit(X_train, y_train)

preds = gauss.predict(X_valid)

NB_acc = accuracy_score(preds, y_valid)

print(NB_acc)
svc = SVC()

svc.fit(X_train, y_train)

preds = svc.predict(X_valid)

svc_acc = accuracy_score(preds, y_valid)

print(svc_acc)
perc = Perceptron()

perc.fit(X_train, y_train)

preds = perc.predict(X_valid)

perc_acc = accuracy_score(preds, y_valid)

print(perc_acc)
dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

preds = dtc.predict(X_valid)

dtc_acc = accuracy_score(preds, y_valid)

print(dtc_acc)
rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

preds = rfc.predict(X_valid)

rfc_acc = accuracy_score(preds, y_valid)

print(rfc_acc)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

preds = knn.predict(X_valid)

knn_acc = accuracy_score(preds, y_valid)

print(knn_acc)
sgd = SGDClassifier()

sgd.fit(X_train, y_train)

preds = sgd.predict(X_valid)

sgd_acc = accuracy_score(preds, y_valid)

print(sgd_acc)
xgb = GradientBoostingClassifier()

xgb.fit(X_train, y_train)

preds = xgb.predict(X_valid)

xgb_acc = accuracy_score(preds, y_valid)

print(xgb_acc)
models = pd.Series(['LogisticRegression', 'GaussianNB', 'SVM', 'Perceptron',

                   'DecisionTree', 'RandomForest', 'KNN', 'SGDClassifier', 'GradientBoostingClassifier'])

accuracies = pd.Series([lr_acc, NB_acc, svc_acc, perc_acc,

                       dtc_acc, rfc_acc, knn_acc, sgd_acc, xgb_acc])

scores = pd.DataFrame({'Model':models, 'Accuracies':accuracies}).sort_values(['Accuracies'], ascending=False)

scores
from sklearn.model_selection import RandomizedSearchCV
# GradientBoosting hyperparameter tuning

learning_rates = [0.001, 0.01, 0.1, 1]

n_estimators = [100, 250, 500 ,1000]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]

max_depth = [5,10,15,20]



params = {'learning_rate':learning_rates,

         'n_estimators':n_estimators,

         'min_samples_split':min_samples_split,

         'min_samples_leaf':min_samples_leaf,

         'max_depth':max_depth}





gbc = GradientBoostingClassifier()

grid_search = RandomizedSearchCV(estimator=gbc, param_distributions=params, scoring='accuracy', n_iter=10,

                                 cv=5, verbose=2, random_state=42, n_jobs=4)

grid_search.fit(X_train, y_train)

print(grid_search.best_score_)

print(grid_search.best_params_)
criterion=['gini', 'entropy']

n_estimators = [100, 250, 500 ,1000]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]

max_depth = [5,10,15,20]



params = {'n_estimators':n_estimators,

         'min_samples_split':min_samples_split,

         'min_samples_leaf':min_samples_leaf,

         'max_depth':max_depth,

         'criterion':criterion}





rfc = RandomForestClassifier()

grid_search = RandomizedSearchCV(estimator=rfc, param_distributions=params, scoring='accuracy', n_iter=10,

                                 cv=5, verbose=2, random_state=42, n_jobs=4)

grid_search.fit(X_train, y_train)

print(grid_search.best_score_)

print(grid_search.best_params_)
n_neighbors = [5,8,11,14]





params = {'n_neighbors':n_neighbors}





knn = KNeighborsClassifier()

grid_search = RandomizedSearchCV(estimator=knn, param_distributions=params, scoring='accuracy', n_iter=10,

                                 cv=5, verbose=2, random_state=42, n_jobs=4)

grid_search.fit(X_train, y_train)

print(grid_search.best_score_)

print(grid_search.best_params_)
# Use our best results from each model on the validation set

best_gbc = GradientBoostingClassifier(n_estimators=500, min_samples_split=5,

                                 min_samples_leaf=10, max_depth=10, learning_rate=0.01)

best_rfc = RandomForestClassifier(n_estimators= 500, min_samples_split= 15, min_samples_leaf=1,

                                  max_depth= 5, criterion= 'gini')

best_knn = KNeighborsClassifier(n_neighbors=11)



models = [best_gbc, best_rfc, best_knn]

for model in models:

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    print(f'Accuracy = {accuracy_score(y_valid, preds)}')
# Train on all training set

best_gbc.fit(X, y)

ids = test['PassengerId']

preds = best_gbc.predict(test.drop('PassengerId', axis=1))

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': preds })

output.to_csv('submission3.csv', index=False)