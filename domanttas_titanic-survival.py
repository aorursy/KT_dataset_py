import numpy as np
import pandas as pd
import os
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
train['label'] = 'train'
test['label'] = 'test'

y = train.Survived

passengerIds = test['PassengerId']

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'label']

X = train[features]
X_test = test[features]

concat_data = pd.concat([X, X_test])
concat_data = pd.get_dummies(concat_data, columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin'])

X_encoded = concat_data[concat_data['label'] == 'train']
X_test_encoded = concat_data[concat_data['label'] == 'test']

X_encoded = X_encoded.drop('label', axis = 1)
X_test_encoded = X_test_encoded.drop('label', axis = 1)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
X_encoded = imputer.fit_transform(X_encoded)
X_test_encoded = imputer.fit_transform(X_test_encoded)
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X_encoded, y, random_state = 0)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

model = GradientBoostingClassifier()
model.fit(train_X, train_y)

validation = model.predict(val_X)
print('MAE:', mean_absolute_error(val_y, validation))
print('Accuracy:', accuracy_score(validation, val_y) * 100)
predicted = model.predict(X_test_encoded)

submission = pd.DataFrame({'PassengerId' : passengerIds, 'Survived' : predicted})
submission.to_csv('submission.csv', index = False)