import numpy as np
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#check the first rows of train data set (to check if the import was Ok)
train.tail()
#check the first rows of test data set (to check if the import was Ok)
test.tail()
# Train data set
train.loc[train['Sex'] == 'male', 'Sex'] = 1
train.loc[train['Sex'] == 'female', 'Sex'] = 0
train.loc[train['Embarked'] == 'S', 'Embarked'] = 0
train.loc[train['Embarked'] == 'C', 'Embarked'] = 0
train.loc[train['Embarked'] == 'Q', 'Embarked'] = 0

# Test data set
test.loc[test['Sex'] == 'male', 'Sex'] = 1
test.loc[test['Sex'] == 'female', 'Sex'] = 0
test.loc[test['Embarked'] == 'S', 'Embarked'] = 0
test.loc[test['Embarked'] == 'C', 'Embarked'] = 0
test.loc[test['Embarked'] == 'Q', 'Embarked'] = 0
# Train data set
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(1)
train['Fare'] = train['Fare'].fillna(train['Fare'].mean())

# Test data set
test['Age'] = test['Age'].fillna(train['Age'].median())
test['Embarked'] = test['Embarked'].fillna(1)
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
predictors = ["Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
dtree = DecisionTreeRegressor(random_state=1)

# Fit model
dtree.fit(train[predictors], train['Survived'])
predictions = dtree.predict(test[predictors])

# convert results to binary
n = 0
for i in predictions:
    if(i > 0.5):
        predictions[n] = 1
    else:
        predictions[n] = 0
    n += 1

# change type of the predictions array to integer
predictions = predictions.astype(int)
predictions.dtype
#Confusion matrix
#from sklearn.metrics import confusion_matrix
#confusion_matrix(predictions, train['Survived'])
# Accuracy
#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(train['Survived'], predictions)
#accuracy
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv("dtree_submit2.csv", index=False)