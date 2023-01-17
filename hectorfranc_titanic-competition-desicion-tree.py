import pandas as pd

import numpy as np

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn import metrics
train_df = pd.read_csv('../input/titanic/train.csv')

test_df = pd.read_csv('../input/titanic/test.csv')

train_df.head()
data_train = train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)

data_test = test_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
categorical_cols = [cname for cname in data_train.columns if

                    data_train[cname].nunique() <= 10 and

                    data_train[cname].dtype == 'object'

                   ]

numerical_cols = [cname for cname in data_train.columns if

                    data_train[cname].dtype in ['int64', 'float64']

                 ]
categorical_cols, numerical_cols
data_train.Age = data_train['Age'].fillna(data_train['Age'].median())

data_test.Age = data_test['Age'].fillna(data_test['Age'].median())
data_test.Fare = data_test['Fare'].fillna(data_test['Fare'].median())
data_train.dropna(subset=['Embarked'], inplace=True)
data_train = pd.get_dummies(data_train)

data_test = pd.get_dummies(data_test)
data_train.columns
x_train, x_test, y_train, y_test = train_test_split(data_train.drop('Survived', axis=1), data_train['Survived'], test_size=0.2)
model = tree.DecisionTreeClassifier()

model.fit(x_train, y_train)
model.score(x_test, y_test) # Accuracy score
output = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':model.predict(data_test)})

output
output.to_csv('my_submission.csv', index=False)
from io import StringIO 

import pydotplus



out = StringIO()

tree.export_graphviz(model, out_file=out)



graph = pydotplus.graph_from_dot_data(out.getvalue())

graph.write_png('titanic.png')