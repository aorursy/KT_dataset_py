import numpy as np

import pandas as pd 

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import xgboost as xgb

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.info()
train.describe()
train.isna().sum()
NUM = ['Age', 'SibSp', 'Parch', 'Fare']

dfs = ['train', 'test']



fig, ax = plt.subplots(4, 2, figsize= (18,30))

for j, df in enumerate(dfs):

    for i, val in enumerate(NUM):

        ax[i, j].hist(eval(df)[val])

        ax[i, j].set_title(f'''{df}: {val}''', fontsize="xx-large");
survivors = train.loc[train['Survived'] > 0, :]

dead = train.loc[train['Survived'] == 0, :]

CAT = ['Survived', 'Pclass', 'Sex', 'Embarked']

dfs = ['dead', 'survivors'] 



fig, ax = plt.subplots(4, 2, figsize= (18,30))



for j, df in enumerate(dfs):

    for i, val in enumerate(NUM):

        ax[i, j].hist(eval(df)[val])

        ax[i, j].set_title(f'''{df}: {val}''', fontsize="xx-large");
numerical = train.loc[:, NUM]

sns.heatmap(numerical.corr(), annot=True, fmt=".2f", square=True, vmin=-1, vmax=1)

plt.show()
train.sample(10)
CAT = ['Survived', 'Pclass', 'Sex', 'Embarked']



fig, ax = plt.subplots(4, 2, figsize=(18,30))



for j, df in enumerate(dfs):

    for i, val in enumerate(CAT):

        eval(df)[val].value_counts(normalize=True).plot(kind='bar', ax=ax[i, j])

        ax[i, j].set_title(val, fontsize="xx-large");
for i, val in enumerate(CAT):

    sns.catplot(x=val, kind='count',hue='Survived',data=train)
X = train.drop(columns=['Survived','Ticket', 'Name', 'PassengerId', 'Cabin'])

X_test = test.drop(columns=['Ticket', 'Name', 'PassengerId', 'Cabin'])

y = train.loc[:, 'Survived']

le = LabelEncoder()

le.fit(X['Sex'])

X['Sex'] = le.transform(X['Sex'])

X_test['Sex'] = le.transform(X_test['Sex'])
X['total_family'] = X['SibSp'] + X['Parch'] + 1

X_test['total_family'] = X_test['SibSp'] + X_test['Parch'] + 1

X['solo'] = np.where(X['total_family'] == 1, True, False)

X_test['solo'] = np.where(X_test['total_family'] == 1, True, False)
X['Embarked'].fillna('S', inplace=True)

X_test['Embarked'].fillna('S', inplace=True)

X['Embarked'] = X['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

X_test['Embarked'] = X_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
X_Embarked = X["Embarked"].values.reshape(-1,1)

X_test_Embarked = X_test["Embarked"].values.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)

X_ohe = ohe.fit_transform(X_Embarked)

X_test_ohe = ohe.fit_transform(X_test_Embarked)

X["EmbarkedS"] = X_ohe[:,0]

X["EmbarkedC"] = X_ohe[:,1]

X["EmbarkedQ"] = X_ohe[:,2]

X_test["EmbarkedS"] = X_test_ohe[:,0]

X_test["EmbarkedC"] = X_test_ohe[:,1]

X_test["EmbarkedQ"] = X_test_ohe[:,2]

X.drop(columns='Embarked', inplace=True)

X_test.drop(columns='Embarked', inplace=True)
X.columns
xgb_model = xgb.XGBClassifier(n_jobs=-1)



parameters = {

        'num_boost_round': [100, 250, 500],

        'eta': [0.05, 0.1, 0.3],

        'max_depth': [6, 9, 12],

        'subsample': [0.9, 1.0],

        'colsample_bytree': [0.9, 1.0],

        'n_estimators': [50, 100, 200]

    }



clf = GridSearchCV(xgb_model, parameters)



clf.fit(X, y)



print(clf.best_score_)

print(clf.best_params_)



clf_best = clf.best_estimator_
kf = KFold(n_splits=3, random_state=42)

for train_index, test_index in kf.split(X):

    xgb_model = clf_best.fit(X.iloc[train_index], y.iloc[train_index])

    predictions = clf_best.predict(X.iloc[test_index])

    print(accuracy_score(y.iloc[test_index], predictions))
clf_best.fit(X,y)

pred = clf_best.predict(X_test)



sub = pd.DataFrame({'PassengerId': test.iloc[:,0], 'Survived': pred})

sub.to_csv("submission.csv", index=False)