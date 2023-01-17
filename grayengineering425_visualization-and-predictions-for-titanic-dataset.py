# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')
train.head()
len(train[train['Survived'] == 1])
len(train[train['Survived'] == 0])
plt.figure(figsize=(10,10))
sns.countplot(x = train.Survived, hue = 'Sex', data = train)
plt.figure(figsize=(20,20))
sns.factorplot("Pclass", "Survived", "Sex", data=train, kind="bar", palette="muted", legend=False)
train['Embarked'].value_counts()
plt.figure(figsize=(10,10))
sns.swarmplot(x='Sex', y='Age', hue='Survived', data=train)
fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(train.Fare[train.Survived == 0] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Fare'] , color='g',shade=True, label='survived')
plt.title('Passenger Fare Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Fare", fontsize = 15)
fig = plt.figure(figsize=(15,8),)

ax=sns.kdeplot(train.Age[train.Survived == 0] , color='gray',shade=True,label='not survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'] , color='g',shade=True, label='survived')
plt.title('Passenger Age Distribution - Survived vs Non-Survived', fontsize = 25)
plt.ylabel("Frequency of Passenger Survived", fontsize = 15)
plt.xlabel("Passenger Age", fontsize = 15)
train.head()
from sklearn.preprocessing import Imputer
from fancyimpute import KNN
categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns
train_num = train[numerical_features]
train_num_matrix = train_num.as_matrix()
train_cat = train[categorical_features]

train_num_filled = pd.DataFrame(KNN(3).complete(train_num_matrix))
train_num_filled.columns = train_num.columns
train_num_filled.index = train_num.index

train = pd.concat([train_num_filled, train_cat], axis = 1)

categorical_features = test.select_dtypes(include = ["object"]).columns
numerical_features = test.select_dtypes(exclude = ["object"]).columns
test_num = test[numerical_features]
test_num_matrix = test_num.as_matrix()
test_cat = test[categorical_features]

test_num_filled = pd.DataFrame(KNN(3).complete(test_num_matrix))
test_num_filled.columns = test_num.columns
test_num_filled.index = test_num.index

test = pd.concat([test_num_filled, test_cat], axis = 1)
train.head()
import re

def getSurname(x, index):
    delimiters = ", ", " "
    regexPattern = '|'.join(map(re.escape, delimiters))
    split_x = re.split(regexPattern, x)
    surname = split_x[index]
    if index == 1:
        if surname.endswith('.'):
            return surname
        else:
            return 'None'
    else:
        return surname

#train['SurName'] = train['Name'].apply(lambda x : getSurname(x, 0))
train['Title'  ] = train['Name'].apply(lambda x : getSurname(x, 1))

#train.loc[(train['Title'] != 'Mr.') && (train['Title'] != 'Mrs.') && (train['Title'] != 'Miss.') && (train['Title'] != 'Master.') && (train['Title'] != 'Dr.') && (train['Title'] != 'Rev.')
#          && (train['Title'] != 'Rev.'), 'HasCabin'] = 'None'
train['Title'].value_counts()
train.loc[:, 'Cabin'] = train.loc[:, 'Cabin'].fillna('None')
test.loc[:, 'Cabin'] = test.loc[:, 'Cabin'].fillna('None')
train = train.drop(train.loc[train['Embarked' ].isnull()].index)
train['Sex'] = train['Sex'].replace({'male' : 0, 'female' : 1})

test['Sex'] = test['Sex'].replace({'male' : 0, 'female' : 1})
train['HasCabin'] = pd.Series(len(train['Cabin']), index=train.index)
train['HasCabin'] = 0
train.loc[train['Cabin'] != 'None', 'HasCabin'] = 1
train['Cabin'] = train['Cabin'].apply(lambda x : x[0] if x != 'None' else 'None')

test['HasCabin'] = pd.Series(len(test['Cabin']), index=test.index)
test['HasCabin'] = 0
test.loc[test['Cabin'] != 'None', 'HasCabin'] = 1
test['Cabin'] = test['Cabin'].apply(lambda x : x[0] if x != 'None' else 'None')
train['Cabin'].value_counts()
train['HasFamily'] = pd.Series(len(train['Parch']), index = train.index)
train['HasFamily'] = 0
train.loc[(train['Parch'] > 0) | (train['SibSp'] > 0), 'HasFamily'] = 1

test['HasFamily'] = pd.Series(len(test['Parch']), index = test.index)
test['HasFamily'] = 0
test.loc[(test['Parch'] > 0) | (test['SibSp'] > 0), 'HasFamily'] = 1
def replaceAge(x):
    if x < 10:
        return 'child'
    elif x < 20:
        return 'teenager'
    elif x < 40:
        return 'young_adult'
    elif x < 60:
        return 'adult'
    else:
        return 'old'

train['Age'] = train['Age'].apply(lambda x : replaceAge(x))
test['Age'] = test['Age'].apply(lambda x : replaceAge(x))
train['FamilySize'] = train['Parch'] + train['SibSp'] + 1
test['FamilySize'] = test['Parch'] + test['SibSp'] + 1
train = train.drop(['Ticket'], axis=1)
test  = test.drop(['Ticket'], axis=1)

train = train.drop(['Name'], axis=1)
test  = test.drop(['Name'], axis=1)
train.head()
corrmat = train.corr()
cols = corrmat.nlargest(10, 'Survived')['Survived'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.2f',
                annot_kws={'size': 10},
                yticklabels=cols.values,
                xticklabels=cols.values)
plt.show()
categorical_features = train.select_dtypes(include = ["object"]).columns
numerical_features = train.select_dtypes(exclude = ["object"]).columns
train_num = train[numerical_features]
train_cat = train[categorical_features]

train_cat = pd.get_dummies(train_cat)
train = pd.concat([train_num, train_cat], axis = 1)

y = train['Survived']
train = train.drop(['Survived'], axis=1)
categorical_features = test.select_dtypes(include = ["object"]).columns
numerical_features = test.select_dtypes(exclude = ["object"]).columns
test_num = test[numerical_features]
test_cat = test[categorical_features]

test_cat = pd.get_dummies(test_cat)
test = pd.concat([test_num, test_cat], axis = 1)
test_names = test.columns.get_values().tolist()
train_names = train.columns.get_values().tolist()

diff_set = list(set(train_names) - set(test_names))

for diff in diff_set:
    test[diff] = 0
train.sort_index(axis=1, inplace=True)
test.sort_index(axis=1,inplace=True)
train.shape
test.shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state=0)
logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
logreg_accy = round(accuracy_score(y_pred, y_test),3)
print(logreg_accy)
C_vals = [0.0001, 0.001, 0.01, 0.1,0.13,0.2, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 4.0,4.5,5.0,5.1,5.5,6.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']

param = {'penalty': penalties, 'C': C_vals, }

grid = GridSearchCV(logreg, param,verbose=False, cv = StratifiedKFold(n_splits=5,random_state=15,shuffle=True), n_jobs=1)

grid.fit(x_train,y_train)

logreg_grid = LogisticRegression(penalty=grid.best_params_['penalty'], C=grid.best_params_['C'])
logreg_grid.fit(x_train,y_train)
y_pred = logreg_grid.predict(x_test)
logreg_accy = round(accuracy_score(y_test, y_pred), 3)
print (logreg_accy)
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-pastel')
y_score = logreg_grid.decision_function(x_test)

FPR, TPR, _ = roc_curve(y_test, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[11,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('ROC for Titanic survivors', fontsize= 18)
plt.show()
from xgboost import XGBClassifier

XGBClassifier = XGBClassifier()
XGBClassifier.fit(x_train, y_train)
y_pred = XGBClassifier.predict(x_test)
XGBClassifier_accy = round(accuracy_score(y_pred, y_test), 3)
print(XGBClassifier_accy)
clf1 = DecisionTreeClassifier(max_depth=6)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3), ['lr', logreg_grid], ['xb', XGBClassifier]],
                        voting='soft', weights=[2, 1, 2, 2, 2])



clf1.fit(x_train, y_train)
clf2.fit(x_train, y_train)
clf3.fit(x_train, y_train)
eclf.fit(x_train, y_train)

y_pred = eclf.predict(x_test)
print(round(accuracy_score(y_pred, y_test)))
## Plotting decision regions
#x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
#y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                     np.arange(y_min, y_max, 0.1))
#
#f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
#
#for idx, clf, tt in zip(product([0, 1], [0, 1]),
#                        [clf1, clf2, clf3, eclf],
#                        ['Decision Tree (depth=4)', 'KNN (k=7)',
#                         'Kernel SVM', 'Soft Voting']):
#
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#
#    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
#    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
#                                  s=20, edgecolor='k')
#    axarr[idx[0], idx[1]].set_title(tt)
#
#plt.show()
y_pred_sub = XGBClassifier.predict(test)
col = ['Survived']

ids = test['PassengerId']
ids = ids.astype(int)

submission = pd.DataFrame(index = ids, columns=col)
submission['Survived'] = y_pred_sub
submission['Survived'] = submission['Survived'].astype(int)
submission.to_csv('titanic_sub.csv')
