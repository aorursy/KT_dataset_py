import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



# magic word for producing visualizations in notebook

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.drop(['Name'], axis=1, inplace=True)

train.drop(['Cabin'], axis=1, inplace=True) # too many null values

train.drop(['Ticket'], axis=1, inplace=True) # drop for now



test.drop(['Name'], axis=1, inplace=True)

test.drop(['Cabin'], axis=1, inplace=True) # too many null values

test.drop(['Ticket'], axis=1, inplace=True) # drop for now
train = train[train['Embarked'].notna()]

train = train[train['Fare'] < 300]
train_clean = train.dropna(thresh=train.shape[1]-1)

print(str(train.shape[0]-train_clean.shape[0])+' rows deleted in train')

train = train_clean
ids = train.pop(['PassengerId'], axis=1, inplace=True)



ids_test = test.pop(['PassengerId'], axis=1, inplace=True)
labels = train.pop(['Survived'], axis=1, inplace=True)
train['Sex'] = pd.factorize(train['Sex'])[0]

train['Embarked'] = pd.factorize(train['Embarked'])[0]



test['Sex'] = pd.factorize(test['Sex'])[0]

test['Embarked'] = pd.factorize(test['Embarked'])[0]
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')



imp_mean.fit(train[['Age']])

train['Age'] = imp_mean.transform(train[['Age']]).ravel()



imp_mean.fit(train[['Fare']])







test['Age'] = imp_mean.transform(test[['Age']]).ravel()

test['Fare'] = imp_mean.transform(test[['Fare']]).ravel()
dummy_columns = ['Sex', 'Pclass', 'Embarked']

for column in dummy_columns:

    just_dummies = pd.get_dummies(train[column])

    train = pd.concat([train, just_dummies], axis=1)      

    train = train.drop([column], axis=1)



for column in dummy_columns:

    just_dummies = pd.get_dummies(test[column])

    test = pd.concat([test, just_dummies], axis=1)      

    test = test.drop([column], axis=1)
scalerStd = StandardScaler()

scalerStd.fit(train)

scalerStd.transform(train)
train, X_test, labels, y_test = train_test_split(train, labels, test_size=0.25, random_state=42)
clf = RandomForestClassifier(max_depth=25,

                             random_state=42,

                             min_samples_leaf=5,

                             n_estimators=25

                            )



clf.fit(train, labels)
scores = cross_val_score(clf, train, labels, cv=5)

scores.mean()
scores = clf.predict(test)
len(scores)
result = pd.DataFrame()

result['PassengerId'] = ids_test

result['Survived'] = scores
result.to_csv('Titanic-results.csv', index=False, header=True)