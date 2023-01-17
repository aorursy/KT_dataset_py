import pandas as pd

from sklearn.feature_extraction import DictVectorizer

from sklearn import svm



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
features = ['Sex', 'Pclass', 'Fare', 'Age', 'Embarked', 'SibSp', 'Parch']

data = train[features]

X = test[features]

target = train['Survived']
data.info()

X.info()
data['Embarked'].fillna('S', inplace=True)

X['Embarked'].fillna('S', inplace=True)

data['Age'].fillna(data['Age'].mean(), inplace=True)

X['Age'].fillna(X['Age'].mean(), inplace=True)

data['Fare'].fillna(data['Fare'].mean(), inplace=True)

X['Fare'].fillna(X['Fare'].mean(), inplace=True)
data.info()

X.info()
dict_vec = DictVectorizer(sparse = False)

data = dict_vec.fit_transform(data.to_dict(orient = 'record'))

dict_vec.feature_names_
X = dict_vec.fit_transform(X.to_dict(orient = 'record'))
model = svm.LinearSVC()

model.fit(data, target)
y = model.predict(X)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': y})

submission.to_csv('submission.csv', index = False)