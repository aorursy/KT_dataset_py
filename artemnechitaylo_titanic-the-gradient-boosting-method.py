!pip install mglearn
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import mglearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
train_data['Title'] = train_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace('Mlle', 'Miss')
train_data['Title'] = train_data['Title'].replace('Ms', 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_data['Title'] = train_data['Title'].map(title_mapping)
train_data['Title'] = train_data['Title'].fillna(0)
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
test_data['Title'] = test_data['Title'].map(title_mapping)
test_data['Title'] = test_data['Title'].fillna(0)
train_data.head()
train_data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], axis='columns', inplace=True)
train_data.head()
print('Embarked:')
print(train_data['Embarked'].value_counts())
train_data['Sex'] = train_data['Sex'].map({'male': 1, 'female': 0})
train_data['Embarked'] = train_data['Embarked'].map({'S': 1, 'C': 0, 'Q': 2})
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
test_data['Embarked'] = test_data['Embarked'].map({'S': 1, 'C': 0, 'Q': 2})
y_train = np.array(train_data['Survived']).astype('float32')
train_data.drop(['Survived'], axis='columns', inplace=True)
train_data = train_data.fillna(train_data.mean())
x_train = np.array(train_data).astype('float32')
test_split = x_train.shape[0] - int(x_train.shape[0] * 0.2)
x_test = x_train[test_split:]
y_test = y_train[test_split:]
x_train = x_train[:test_split]
y_train = y_train[:test_split]
def plot_feature_importances(model):
    n_features = x_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), list(train_data))
    plt.xlabel('Importance of the feature')
    plt.ylabel('Feature')
forest = RandomForestClassifier(n_estimators=10, random_state=1)
forest.fit(x_train, y_train)
plot_feature_importances(forest)
print(f'Correctness on the training set(Boost forest): {int(forest.score(x_train, y_train) * 1000) / 10}%')
print(f'Correctness on the test set(Boost forest): {int(forest.score(x_test, y_test) * 1000) / 10}%')
boost_forest = GradientBoostingClassifier(max_depth=5, learning_rate=0.01, n_estimators=200)
boost_forest.fit(x_train, y_train)
plot_feature_importances(boost_forest)
print(f'Correctness on the training set(Boost forest): {int(boost_forest.score(x_train, y_train) * 1000) / 10}%')
print(f'Correctness on the test set(Boost forest): {int(boost_forest.score(x_test, y_test) * 1000) / 10}%')
plt.figure(figsize=(11,11), dpi= 80)
sns.heatmap(train_data.corr(), xticklabels=train_data.corr().columns, yticklabels=train_data.corr().columns, cmap='RdYlGn', center=0, annot=True)

plt.title('Correlogram of surviving', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
# deleting 'Fare'
train_data.drop(['Fare'], axis='columns', inplace=True)
x_train = np.array(train_data).astype('float32')
test_split = x_train.shape[0] - int(x_train.shape[0] * 0.2)
x_test = x_train[test_split:]
x_train = x_train[:test_split]
boost_forest = GradientBoostingClassifier(max_depth=5, learning_rate=0.01, n_estimators=200)
boost_forest.fit(x_train, y_train)
plot_feature_importances(boost_forest)
print(f'Correctness on the training set(Boost forest): {int(boost_forest.score(x_train, y_train) * 1000) / 10}%')
print(f'Correctness on the test set(Boost forest): {int(boost_forest.score(x_test, y_test) * 1000) / 10}%')
test_ids = test_data['PassengerId']
test_data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name', 'Fare'], axis='columns', inplace=True)
test_data = test_data.fillna(test_data.mean())
x_test1 = np.array(test_data).astype('float32')
predictions = boost_forest.predict(x_test1)
submission = pd.DataFrame(data = predictions,columns = ['Survived'])
submission['PassengerId'] = test_ids
submission['Survived'] = submission['Survived'].apply(lambda x : np.int32(x))
submission.to_csv('titanic_sub.csv',index=False)