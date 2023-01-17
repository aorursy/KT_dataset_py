import pandas as pd
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.shape
test_df.shape
train_df.tail()
test_df.tail()
train_df.info()
test_df.info()
from sklearn.metrics import accuracy_score

from sklearn.model_selection import (train_test_split, cross_val_score, KFold,

                                     GridSearchCV)

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier
x = train_df['Sex']
y_pred = x.map({'female': 1, 'male': 0}).astype(int)
y = train_df['Survived']

accuracy_score(y, y_pred)
x_test = test_df['Sex']

y_test_pred = x_test.map({'female': 1, 'male': 0}).astype(int)
submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': y_test_pred

})

submission.to_csv('submission.csv', index=False)
columns = ['Age', 'Pclass', 'Sex', 'Embarked']
X = train_df[columns].copy()

y = train_df['Survived']

X_test = test_df[columns].copy()
X.tail()
X.isnull().sum()
X_test.isnull().sum()
age_mean = X['Age'].mean()

print(f'Age mean: {age_mean}')
X['AgeFill'] = X['Age'].fillna(age_mean)

X_test['AgeFill'] = X_test['Age'].fillna(age_mean)
X = X.drop(['Age'], axis=1)

X_test = X_test.drop(['Age'], axis=1)
embarked_freq = X['Embarked'].mode()[0]

print(f'Embarked freq: {embarked_freq}')
X['EmbarkedFill'] = X['Embarked'].fillna(embarked_freq)

X_test['EmbarkedFill'] = X_test['Embarked'].fillna(embarked_freq)
X = X.drop(['Embarked'], axis=1)

X_test = X_test.drop(['Embarked'], axis=1)
X.isnull().sum()
X_test.isnull().sum()
gender_map = {'female': 0, 'male': 1}

X['Gender'] = X['Sex'].map(gender_map).astype(int)

X_test['Gender'] = X_test['Sex'].map(gender_map).astype(int)
X = X.drop(['Sex'], axis=1)

X_test = X_test.drop(['Sex'], axis=1)
X = pd.get_dummies(X, columns=['EmbarkedFill'])

X_test = pd.get_dummies(X_test, columns=['EmbarkedFill'])
X.tail()
X_train, X_val, y_train, y_val = train_test_split(

    X, y, test_size=0.2, random_state=1)
X_train.tail()
print(f'Num of training samples: {len(X_train)}')

print(f'Num of validation samples: {len(X_val)}')
clf = LogisticRegression(solver='liblinear')

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

y_val_pred = clf.predict(X_val)
print(f'Accuracy on Training Set: {accuracy_score(y_train, y_train_pred):.5f}')

print(f'Accuracy on Validation Set: {accuracy_score(y_val, y_val_pred):.5f}')
X_train, X_val, y_train, y_val = train_test_split(

    X, y, test_size=0.2, random_state=33)

print(f'Num of training samples: {len(X_train)}')

print(f'Num of validation samples: {len(X_val)}')
X_train.tail()
clf = LogisticRegression(solver='liblinear')

clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

y_val_pred = clf.predict(X_val)

print(f'Accuracy on Training Set: {accuracy_score(y_train, y_train_pred):.5f}')

print(f'Accuracy on Validation Set: {accuracy_score(y_val, y_val_pred):.5f}')
def cross_val(clf, X, y, K=5, random_state=0):

    cv = KFold(K, shuffle=True, random_state=random_state)

    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

    return scores
clf = LogisticRegression(solver='liblinear')

scores = cross_val(clf, X, y)

print(f'Scores: {scores}')

print(f'Mean Score: {scores.mean():.5f} (+/-{scores.std()*2:.5f})')
X_train, X_val, y_train, y_val = train_test_split(

    X, y, test_size=0.2, random_state=33)

clf = LogisticRegression(solver='liblinear')

clf.fit(X_train, y_train)

y_test_pred = clf.predict(X_test)

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': y_test_pred

})

#submission.to_csv('submission.csv', index=False)
def print_cross_val_score(scores):

    print(f'Scores: {scores}')

    print(f'Mean Score: {scores.mean():.5f} (+/-{scores.std()*2:.5f})')
clf = DecisionTreeClassifier(

    criterion='entropy', max_depth=2, min_samples_leaf=2)

scores = cross_val(clf, X, y)

print_cross_val_score(scores)
clf = DecisionTreeClassifier(

    criterion='entropy', max_depth=3, min_samples_leaf=2)

scores = cross_val(clf, X, y)

print_cross_val_score(scores)
clf = DecisionTreeClassifier(

    criterion='entropy', max_depth=2, min_samples_leaf=2)



param_grid = {'max_depth': [2, 3, 4, 5], 'min_samples_leaf': [2, 3, 4, 5]}

cv = KFold(5, shuffle=True, random_state=0)



grid_search = GridSearchCV(

    clf, param_grid, cv=cv, n_jobs=-1, verbose=1, return_train_score=True)

grid_search.fit(X, y)
print(f'Scores: {grid_search.best_score_:.5f}')

print(f'Best Parameter Choice: {grid_search.best_params_}')
y_test_pred = grid_search.predict(X_test)

submission = pd.DataFrame({

    'PassengerId': test_df['PassengerId'],

    'Survived': y_test_pred

})

#submission.to_csv('submission.csv', index=False)
clf = LogisticRegression(solver='liblinear')



param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

cv = KFold(5, shuffle=True, random_state=0)



grid_search = GridSearchCV(

    clf, param_grid, cv=cv, n_jobs=-1, verbose=1, return_train_score=True)

grid_search.fit(X, y)
print(f'Scores: {grid_search.best_score_:.5f}')

print(f'Best Parameter Choice: {grid_search.best_params_}')