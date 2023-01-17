import pandas as pd

import numpy as np



from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, GridSearchCV

#from sklearn.preprocessing import StandardScaler
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.shape
train.head()
train.describe(include='all')
train['Age'].value_counts(dropna=False)
train['Sex'].value_counts(dropna=False)
test.shape
test.head()
train.columns.to_list()
#columns = train.columns.to_list()

#columns.remove('Survived')



columns = [#'PassengerId',

         #'Survived',

         'Pclass',

         #'Name',

         'Sex',

         #'Age',

         #'SibSp',

         #'Parch',

         #'Ticket',

         ##'Fare',

         #'Cabin',

         'Embarked']



target = 'Survived'
train[columns].fillna(value=0.0, inplace=True)
test[columns].fillna(value=0.0, inplace=True)
X_train = train[columns]

y_train = train[target]

X_test = test
X_train = pd.get_dummies(X_train, columns=['Sex', 'Embarked'])

X_test = pd.get_dummies(X_test, columns=['Sex', 'Embarked'])
columns_dummies = X_train.columns.to_list()

columns_dummies
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=0)
parameters = {

    'learning_rate': [0.1, 0.2, 0.3, 0.5, 0.7, 1],

    'max_depth':[2, 3, 4],

    'n_estimators':[10, 20, 50, 100],

    'random_state': [0]

}



clf = GridSearchCV(GradientBoostingClassifier(), parameters, n_jobs=-1)



clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))

print(clf.best_params_)



prediction = clf.predict(X_val)



#X_val['Survived'] = prediction
print('Train:      ', clf.score(X_train, y_train))

print('Validation: ', clf.score(X_val, y_val))
confusion_matrix(y_val, prediction)
tn, fp, fn, tp = confusion_matrix(y_val, prediction).ravel()

#(tn, fp, fn, tp)

print('True Negative:  ', tn)

print('False Positive: ', fp)

print('False Negative: ', fn)

print('True Positive:  ', tp)
np.unique(prediction, return_counts=True)
test_prediction = clf.predict(X_test[columns_dummies])
X_test['Survived'] = test_prediction

X_test['Survived'].value_counts()
X_test[['PassengerId', 'Survived']]
X_test[['PassengerId', 'Survived']].to_csv('titanic_prediction.csv', index=False)