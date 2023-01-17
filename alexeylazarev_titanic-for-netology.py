import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import graphviz

%matplotlib inline
data = pd.read_csv('/kaggle/input/train.csv')

data_test = pd.read_csv('/kaggle/input/test.csv')
test_pass_id = data_test.PassengerId

y = data.Survived

data.drop(columns=['Survived'], axis = 1, inplace = True)
def modify_data(data):

    result = data.copy()

    result['NameLength'] = result['Name'].apply(lambda x: len(x))

    result['Cabin'].fillna(value='X', inplace=True)

    result['CabinLetter'] = result['Cabin'].apply(lambda x: str(x)[0])

    result = result.drop(['PassengerId','Ticket','Name', 'Cabin', 'Embarked'], axis=1)

    result = pd.get_dummies(result, columns=['Sex', 'CabinLetter'], drop_first=True)

    result['Age'].fillna(value=result['Age'].mean(),inplace=True)

    result['Fare'].fillna(value=result['Fare'].mean(), inplace=True)

    return result

X = modify_data(data)

X.head()

X.drop(['CabinLetter_T'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits = 6)

#scores = cross_val_score(tree, X_train, y_train, cv = kfold)

#print("Train mean score: {0:.2f}".format(scores.mean()))

#print("Test score: {0:.2f}".format(tree.score(X_test, y_test)))
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[1,2,3,4,5,6,7,8,9]}

clf = GridSearchCV(tree, parameters, cv=kfold)

clf.fit(X, y)
plt.plot(parameters.get('max_depth'),clf.cv_results_.get('mean_test_score'))

tree = DecisionTreeClassifier(max_depth = 6)

tree.fit(X,y)
for n, im in zip (X.columns, tree.feature_importances_):

    print(n,'\t', '{0:.3f}'.format(im))

from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='titanic.dot',

                class_names=['died','survied'],

                feature_names=X.columns,

                filled=True)
import graphviz

with open('titanic.dot') as f:

    graph = f.read()

graphviz.Source(graph)


cross_val_score(tree, X, y, cv=kfold).mean()
X_submit = modify_data(data_test)

y_submit = tree.predict(X_submit)

submit = pd.DataFrame({'PassengerId':test_pass_id}).join(pd.DataFrame({'Survived':y_submit}))
%pwd
submit[['PassengerId', 'Survived']].to_csv('titanic_submit.csv', index=False)