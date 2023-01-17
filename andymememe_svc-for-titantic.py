import pandas as pd

import numpy as np

from sklearn_pandas import DataFrameMapper

from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
# Prepare dataset

train_data = pd.read_csv('../input/train.csv')

train_x = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

train_x = train_x.fillna(train_x.mean())

train_x['Ebk'] = train_x['Embarked'].apply(lambda x: 0 if x == 'C' else (1 if x == 'Q' else 2))

train_y = train_data['Survived'].as_matrix()

test_data = pd.read_csv('../input/test.csv')

test_x = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

test_x = test_x.fillna(test_x.mean())

test_x['Ebk'] = test_x['Embarked'].apply(lambda x: 0 if x == 'C' else (1 if x == 'Q' else 2))

test_id = test_data['PassengerId'].as_matrix()

mapper = DataFrameMapper([('Pclass', None),

                          ('Sex', preprocessing.LabelBinarizer()),

                          ('Age', None),

                          ('SibSp', None),

                          ('Parch', None),

                          ('Fare', None),

                          ('Ebk', None)

                         ], df_out=True)

train_x = mapper.fit_transform(train_x).as_matrix()

test_x = mapper.fit_transform(test_x).as_matrix()
# Prepare model

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svc = SVC()

grid_search = GridSearchCV(svc, parameters)

grid_search.fit(train_x, train_y)
# Doing test

test_y = grid_search.predict(test_x)

test_y = test_y.reshape(-1, 1)

test_id = test_id.reshape(-1, 1)

result = np.hstack((test_id, test_y))

result = pd.DataFrame(result)

result.to_csv('result.csv', header=['PassengerId', 'Survived'], index=False)