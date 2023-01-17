import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import neighbors
train_main = pd.read_csv('../input/train.csv')
test_main = pd.read_csv('../input/test.csv')
# preview data
train_main.head()
# copy
train = train_main.copy()
test = test_main.copy()
# take care of NaN's probably not the best way
train['Cabin'] = train['Cabin'].fillna('NotAssigned')
train['Age'] = train['Age'].fillna(-1.0)
test['Cabin'] = test['Cabin'].fillna('NotAssigned')
test['Age'] = test['Age'].fillna(-1.0)
test['Fare'] = test['Fare'].fillna(-1.0)
train_data = train.drop(columns=['Name','PassengerId', 'Ticket'])
test_data = test.drop(columns=['Name', 'PassengerId', 'Ticket'])
X_train_data = train_data.drop(columns=['Survived'])
Y_train_data = train_data[['Survived']]
X_test_data = test_data.copy()
bins= [ -1.0, 0.0, 1.0, 3.0, 13.0, 20.0, 26.0, 41.0, 61.0, 150.0]
labels = ['Unknown', 'Infant', 'Toddler','Kid','Teen','Young Adult', 'Adult', 'Middle Aged', 'Senior Citizens']

X_train_data['AgeGroup'] = pd.cut(X_train_data['Age'], bins=bins, labels=labels, right=False)
X_test_data['AgeGroup'] = pd.cut(X_test_data['Age'], bins=bins, labels=labels, right=False)

X_train_data = X_train_data.drop(columns=['Age'])
X_test_data = X_test_data.drop(columns=['Age'])
X_test_data['test'] = True
X_train_data['test'] = False
combined_data = pd.concat([X_train_data, X_test_data])
one_hot_encode = pd.get_dummies(combined_data, columns=['Sex', 'Cabin', 'AgeGroup', 'Embarked'])
X_train_data = one_hot_encode[one_hot_encode.test == False]
X_test_data = one_hot_encode[one_hot_encode.test == True]
X_train_data = X_train_data.drop(columns=['test'])
X_test_data = X_test_data.drop(columns=['test'])
X = X_train_data.values
y = Y_train_data.values
# init KNN algo
clf = neighbors.KNeighborsClassifier()
# training model
clf.fit(X, y.ravel())
predicted = clf.predict(X_test_data.values)
expected_y = pd.read_csv('../input/gender_submission.csv')['Survived'].values
from sklearn.metrics import accuracy_score
accuracy_score(expected_y, predicted)
