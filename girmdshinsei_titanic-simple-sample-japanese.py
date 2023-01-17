import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')
data = pd.concat([train,test],axis = 0)
import seaborn as sns

sns.set(style="darkgrid")



data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

train['FamilySize'] = data['FamilySize'][:len(train)]

test['FamilySize'] = data['FamilySize'][len(train):]

sns.countplot(x='FamilySize', data=train, hue='Survived')
data['Family1'] = (data['FamilySize'] == 1)*1
train.info()

print('_'*40)

test.info()
train.describe(include=['O'])
train.describe()
data['Age'].fillna(data['Age'].median(), inplace=True)
data = pd.concat([data,pd.get_dummies(data['Sex'],drop_first=True)],axis=1)

data = data.drop('Sex',axis=1)
train = data.iloc[:len(train),:]

test = data.iloc[len(train):,:]
feature_name = ['male','Age','Family1']
X_train = train.loc[:,feature_name]

y_train = train.loc[:,'Survived']



X_test = test.loc[:,feature_name]
X_train.head(2)
from sklearn.linear_model import LogisticRegression



clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.model_selection import KFold



y_preds = []

models = []

oof_train = np.zeros((len(X_train),))

cv = KFold(n_splits=5, shuffle=True, random_state=0)



for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train)):

    

    X_tr = X_train.loc[train_index, :]

    X_val = X_train.loc[valid_index, :]

    y_tr = y_train[train_index]

    y_val = y_train[valid_index]

    

    clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)

    clf.fit(X_tr, y_tr)

    

    oof_train[valid_index] = clf.predict(X_val)

    y_pred = clf.predict(X_test)



    y_preds.append(y_pred)

    models.append(clf)
from sklearn.metrics import accuracy_score



accuracy_score(y_train,oof_train)
sub = pd.DataFrame(pd.read_csv('../input/titanic/test.csv')['PassengerId'])

sub['Survived'] = list(map(int, y_pred))

sub.to_csv('submission.csv', index=False)