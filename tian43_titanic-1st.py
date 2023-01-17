# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_valid_y = train['Survived']
train.info()

test.info()
train.columns.values
train.describe(include = ['O'])
train.describe()
def viewcat(df, cat, target):

    return df[[cat, target]].groupby(cat).mean().sort_values(by = target, ascending = False)
viewcat(train, 'Pclass', 'Survived')
viewcat(train, 'Sex', 'Survived')
viewcat(train, 'SibSp', 'Survived')
viewcat(train, 'Embarked', 'Survived')
def plot_cat(df, cat, target):

    sns.barplot(df[cat], df[target])

    

plot_cat(train, 'Sex', 'Survived')
plot_cat(train, 'Embarked', 'Survived')
def plot_num(df, var, target, **kwargs):

    row = kwargs.get('row', None)

    col = kwargs.get('col',  None)

    grid = sns.FacetGrid(df, hue = target, row = row, col = col)

    grid.map(sns.distplot, var)

    grid.add_legend()



plot_num(train, 'Fare', 'Survived')
grid = sns.FacetGrid(train, hue = 'Survived', col = 'Embarked')

grid.map(plt.hist, 'Age', alpha = .5, bins = 20)

grid.add_legend()
train.drop('Survived', axis = 1, inplace = True)
full = pd.concat([train,test])
full.info()
full.describe(include = ['O'])
full['Embarked'].fillna('S', inplace = True)
full.info()
Sex = pd.get_dummies(full['Sex'])
Embarked = pd.get_dummies(full['Embarked'])
full['Title'] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

pd.crosstab(full['Title'], full['Sex'])
full['Title'] = full['Title'].replace(['Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'the Countess'], 'Rare')

full['Title'] = full['Title'].replace(['Mlle', 'Mme', 'Ms'], 'Mrs')

full['Title'].unique()

Title = pd.get_dummies(full['Title'])
Title
full.drop(['Title', 'Embarked', 'Name', 'Sex', 'PassengerId', 'Cabin'], axis = 1, inplace = True)
full
full['Family'] = full['SibSp'] + full['Parch']
full.info()
full['Fare'] = full['Fare'].fillna(full['Fare'].mean())
full.info()
full.drop(['SibSp', 'Parch'], inplace = True, axis = 1)
full = pd.concat([full,Sex, Embarked, Title], axis = 1)
full
sns.heatmap(full.corr())
for i in range(3):

    guess = full[full['Pclass'] == i+1]['Age'].median()

    full.loc[(full['Age'].isnull()) & (full['Pclass'] == i+1), 'Age'] = guess
full.info()
full.drop('Ticket', axis = 1, inplace = True)
full
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression  

from sklearn.naive_bayes import GaussianNB  

from sklearn.metrics import accuracy_score
train_valid_x = full[0:891]

test_x = full[891:]

X_train, X_valid, y_train, y_valid = train_test_split(train_valid_x, train_valid_y, random_state = 0) 
X_train.info()
scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_valid_scaled = scaler.transform(X_valid)

X_train_valid_scaled = scaler.transform(train_valid_x)

y_train_valid = train_valid_y
class Model(object):

    def __init__(self, clf, seed = 0, **params):

        params['random_state'] = seed

        self.clf = clf(**params)

    def fit(self, x, y):

        return self.clf.fit(x, y)

    def predict(self, x):

        return self.clf.predict(x)

    def score(self, x, y):

        y_predict = self.clf.predict(x)

        return accuracy_score(y, y_predict)

    

        

        
rf = Model(clf = RandomForestClassifier)

gb = Model(clf = GradientBoostingClassifier)

mlp = Model(clf = MLPClassifier)

svc = Model(clf = SVC)

xgb = Model(clf = XGBClassifier)

clf1 = LogisticRegression(random_state=0)  

clf2 = RandomForestClassifier(random_state=0)  

clf3 = GaussianNB()

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard') 

eclf.fit(X_train_scaled, y_train)

rf.fit(X_train_scaled, y_train)

gb.fit(X_train_scaled, y_train)

mlp.fit(X_train_scaled, y_train)

svc.fit(X_train_scaled, y_train)

xgb.fit(X_train_scaled, y_train)
from sklearn.model_selection import cross_val_score

rf_scores = cross_val_score(RandomForestClassifier(), 

                            X_train_valid_scaled, y_train_valid, scoring = 'accuracy',

                            cv= 5)

gb_scores = cross_val_score(GradientBoostingClassifier(), X_train_valid_scaled, y_train_valid, scoring = 'accuracy'

                           ,cv = 5)

#mlp_scores = cross_val_score(MLPClassifier(), X_train_valid_scaled, y_train_valid, scoring = 'accuracy'

                           #, cv = 3)

svc_scores = cross_val_score(SVC(), X_train_valid_scaled, y_train_valid, scoring = 'accuracy'

                           , cv = 5)

xgb_scores = cross_val_score(XGBClassifier(), X_train_valid_scaled, y_train_valid, scoring = 'accuracy'

                           , cv = 5)

elf_scores = cross_val_score(eclf, X_train_valid_scaled, y_train_valid, scoring = 'accuracy'

                           , cv = 5)
rf_scores.mean()
gb_scores.mean()
svc_scores.mean()
xgb_scores.mean()
elf_scores.mean()
rf_score = rf.score(X_valid_scaled, y_valid)

gb_score = gb.score(X_valid_scaled, y_valid)

mlp_score = mlp.score(X_valid_scaled, y_valid)

svc_score = svc.score(X_valid_scaled, y_valid)

xgb_score = xgb.score(X_valid_scaled, y_valid)

elf_y_predict = eclf.predict(X_valid_scaled)

elf_score = accuracy_score(y_valid, elf_y_predict)
print(rf_score, gb_score, mlp_score, svc_score, xgb_score, elf_score)
X_test_scaled = scaler.transform(test_x)
y_predict = xgb.predict(X_test_scaled)
y_predict
PassengerId = test['PassengerId']

submission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': y_predict})
submission.to_csv('submission.csv', index = False)