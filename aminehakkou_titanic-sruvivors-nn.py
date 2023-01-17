import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import Imputer
orig_train_data = pd.read_csv('../input/train.csv')

orig_test_data = pd.read_csv('../input/test.csv')
def clean(data):

    data = data.loc[:, ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

    # associate 1 to male & -1 to female

    data.loc[data.Sex == 'male', 'Sex'] = 1

    data.loc[data.Sex == 'female', 'Sex'] = -1

    

    # associate 1 to C 0 to Q & -1 to S

    data.loc[data.Embarked == 'C', 'Embarked'] = 1

    data.loc[data.Embarked == 'Q', 'Embarked'] = 0

    data.loc[data.Embarked == 'S', 'Embarked'] = -1

    

    return data



## clean training data

train_data = clean(orig_train_data)

## clean test data

test_data = clean(orig_test_data)



train_data.head()
# convert to matrix

imputer = Imputer()

Y = train_data.loc[:, 'Survived'].values

X = imputer.fit_transform(train_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])

X_test = imputer.fit_transform(test_data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])
# define model

clf = MLPClassifier(solver='adam', alpha=0.001, hidden_layer_sizes=(5, 2), random_state=1)



# train model

clf.fit(X, Y)



# test model

Y_test = clf.predict(X_test)

orig_test_data.loc[:, 'Survived'] = Y_test

orig_test_data.loc[:, ['PassengerId', 'Survived']].to_csv('predictions.csv', header=True, index=False)
