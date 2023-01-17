# Standard imports



import numpy as np

import pandas as pd

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler

# Load in train & test data



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
# Drop feature columns that may have small effect on output. Can optimize later.

# Create X_train, y_train, X_test dataframes by dropping appropriate columns



X_train = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'],1)

y_train = train_data['Survived']

X_test = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],1)
# Clean 'age' columns by replacing missing values with median of X_train['Age']



# Clean X_train 'age' data by filling missing data with median of 'age' column

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())



# Clean X_test 'age' column data as above

X_test['Age'] = X_test['Age'].fillna(X_train['Age'].median())
# Converting strings in 'Sex' column to 0, 1 for training & test data



X_train.loc[X_train['Sex'] == 'male', 'Sex'] = 0

X_train.loc[X_train['Sex'] == 'female', 'Sex'] = 1



X_test.loc[X_test['Sex'] == 'male', 'Sex'] = 0

X_test.loc[X_test['Sex'] == 'female', 'Sex'] = 1



# Must convert dtype to int



X_train['Sex'] = X_train['Sex'].astype(int)

X_test['Sex'] = X_test['Sex'].astype(int)



# Clean 'Fare' column by replacing missing data with column median



X_test['Fare'] = X_test['Fare'].fillna(X_test['Fare'].median())
# Convert 'Embarked' S, C, Q strings to ints 0, 1, 2 for both feature datasets



X_train.loc[X_train['Embarked'] == 'S', 'Embarked'] = 0

X_train.loc[X_train['Embarked'] == 'C', 'Embarked'] = 1

X_train.loc[X_train['Embarked'] == 'Q', 'Embarked'] = 2



X_test.loc[X_test['Embarked'] == 'S', 'Embarked'] = 0

X_test.loc[X_test['Embarked'] == 'C', 'Embarked'] = 1

X_test.loc[X_test['Embarked'] == 'Q', 'Embarked'] = 2



# Convert object types to floats



X_train['Embarked'] = X_train['Embarked'].astype('float')

X_test['Embarked'] = X_test['Embarked'].astype('float')



# Fill missing values with median value



X_train['Embarked'].fillna(X_train['Embarked'].median(), inplace=True)
# Replace Sibsp and Parch columns with combined Family column



X_train['Family'] = X_train['SibSp'] + X_train['Parch']



X_test['Family'] = X_test['SibSp'] + X_test['Parch']
# Set training data to X, y and convert to np arrays



X = np.array(X_train)

y = np.array(y_train)



X_test = np.array(X_test)



# Fit standard scaler to training data then apply to test data



scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)
# Set classifier type & parameters

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,

                    hidden_layer_sizes=(25,10,5), random_state=1, max_iter=500, shuffle=True)



# Fit classifier

clf.fit(X, y)



# Print score

score = clf.score(X,y)

print(score)
# Run classifier on test data



y_test = clf.predict(X_test)



score = clf.score(X,y)



print(y_test)

print(score)
submission = pd.DataFrame({

    'PassengerId': test_data['PassengerId'],

    'Survived': y_test

})



submission.to_csv('kaggle.csv', index=False)