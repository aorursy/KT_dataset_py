import os

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics



train_data_file = os.path.join('/kaggle/input/titanic/', 'train.csv')

test_data_file = os.path.join('/kaggle/input/titanic/', 'test.csv')



train_df = pd.read_csv(train_data_file)

test_df = pd.read_csv(test_data_file)
print("Columns Overview")

print(train_df.columns)



print("Interesting Cross Tables")

print(pd.crosstab(train_df.Survived, train_df.Sex))

print(pd.crosstab(train_df.Survived, train_df.Age))

print(pd.crosstab(train_df.Survived, train_df.Pclass))

print(pd.crosstab(train_df.Survived, train_df.SibSp))

print(pd.crosstab(train_df.Survived, train_df.Parch))

print(pd.crosstab(train_df.Survived, train_df.Fare))

print(pd.crosstab(train_df.Survived, train_df.Embarked))



print("General Description")

print(train_df.describe())



with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):

    print("Covariance Matrix")

    print(train_df.cov())

    print("Correlation Matrix")

    print(train_df.corr())
sex_mapping = {'male': 1, 'female': 2,}

embark_mapping = {'C': 1, 'S': 2, 'Q': 3}



train_df = train_df.replace({'Sex': sex_mapping, 'Embarked': embark_mapping})

test_df = test_df.replace({'Sex': sex_mapping, 'Embarked': embark_mapping})
train_df = train_df.filter(items=['Survived', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked'])

test_df = test_df.filter(items=['PassengerId', 'Sex', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked'])
train_df.Age.fillna(test_df.Age.mean(), inplace=True)

train_df.Fare.fillna(test_df.Fare.mean(), inplace=True)

train_df.Embarked.fillna(test_df.Embarked.mean(), inplace=True)

test_df.Age.fillna(test_df.Age.mean(), inplace=True)

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)
X = train_df.iloc[:, 1:].values

y = train_df.iloc[:, 0].values



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = RandomForestClassifier(n_estimators=500, random_state=0)

regressor.fit(X_train, y_train)

y_valid_pred = regressor.predict(X_valid)
print("Confusion Matrix")

print(metrics.confusion_matrix(y_valid, y_valid_pred))

print("Classification Report")

print(metrics.classification_report(y_valid, y_valid_pred))

print("Accuracy")

print(metrics.accuracy_score(y_valid, y_valid_pred))
y_test_pred = regressor.predict(test_df.iloc[:, 1:].values)
submission = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_test_pred})

submission.to_csv('submission.csv', index=False)