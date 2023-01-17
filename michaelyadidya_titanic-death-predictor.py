import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
training = pd.read_csv('../input/train.csv')
training.describe()
training['Age'] = training['Age'].fillna(training['Age'].median())
training.describe()
print('-------------Null type in Dataset-----------')
training.isnull().sum()
training.columns
#Dropping Unnecessary Columns

train_data = training[['PassengerId','Pclass','Sex','Age']]

test_labels = training['Survived']
train_data.head()
train_data.isnull().sum()
#train_data['Embarked'] = pd.get_dummies(train_data['Embarked'])
train_data['Sex'] = pd.get_dummies(train_data['Sex'])
train_data.head()
print('Shape of Train_data: ',train_data.shape)
print('Shape of Labels: ',test_labels.shape)
X_train = train_data.head(500)
y_train = test_labels.head(500)

X_test = train_data.tail(391)
y_test = test_labels.tail(391)
model = LogisticRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
confusion_matrix(y_test, predictions)
test_data = pd.read_csv('../input/test.csv')
test_data = test_data[['PassengerId','Pclass','Sex','Age']]
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
#test_data['Embarked'] = pd.get_dummies(test_data['Embarked'])
test_data['Sex'] = pd.get_dummies(test_data['Sex'])
predictions = model.predict(test_data)
predictions
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('submission.csv', index=False)
