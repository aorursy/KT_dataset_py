import pandas as pd

import numpy as np
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

print('Dimension of training data= ', train_data.shape)

train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

print('Dimension of test data= ', test_data.shape)

test_data.head()
women = train_data.loc[train_data.Sex == 'female']['Survived']

print('Total number of women = ', len(women))

print('Number of women survived = ', sum(women))

print('% of women survived = ', sum(women) / len(women) )
men = train_data.loc[train_data.Sex == 'male']['Survived']

print('Total number of men = ', len(men))

print('Number of men survived = ', sum(men))

print('% of men survived = ', sum(men) / len(men) )
from sklearn.ensemble import RandomForestClassifier



y = train_data['Survived']

features = ['Pclass', 'Sex', 'SibSp', 'Parch']

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1 )

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index= False)

print('Submission successfully saved!')