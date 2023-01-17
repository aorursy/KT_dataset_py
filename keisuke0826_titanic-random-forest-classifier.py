import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Read the input datasets

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



# Fill missing numeric values with median for that column

train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

test_data['Fare'].fillna(test_data['Fare'].mean(), inplace=True)



print(train_data.info())

print(test_data.info())
# Encode sex as int 0=female, 1=male

train_data['Sex'] = train_data['Sex'].apply(lambda x: int(x == 'male'))



# Extract pclass, sex, age, fare features

X = train_data[['Pclass', 'Sex', 'Age', 'Fare']].as_matrix()

print(np.shape(X))



# Extract survival target

y = train_data[['Survived']].values.ravel()

print(np.shape(y))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score



# Build the classifier

kf = KFold(n_splits=3)

model = RandomForestClassifier()

scores = [

    model.fit(X[train], y[train]).score(X[test], y[test])

    for train, test in kf.split(X)

]

    

print("Mean 3-fold cross validation accuracy: %s" % np.mean(scores))
# Create model with all training data

classifier = model.fit(X, y)



# Encode sex as int 0=female, 1=male

test_data['Sex'] = test_data['Sex'].apply(lambda x: int(x == 'male'))



# Extract pclass, sex, age, fare features

X_ = test_data[['Pclass', 'Sex', 'Age', 'Fare']].as_matrix()



# Predict if passengers survived using model

y_ = classifier.predict(X_)



# Append the survived attribute to the test data

test_data['Survived'] = y_

predictions = test_data[['PassengerId', 'Survived']]

print(predictions)



# Save the output for submission

predictions.to_csv('submission.csv', index=False)