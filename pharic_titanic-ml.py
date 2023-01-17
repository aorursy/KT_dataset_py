import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')

df.describe()

df.head()
survived = df[df['Survived'] == 1]

dead = df[df['Survived'] == 0]
dead.describe()
survived['Age'].hist()

plt.xlabel('Age')

plt.ylabel('No. of survivors')

plt.show()

dead['Age'].hist()

plt.xlabel('Age')

plt.ylabel('No. of dead')

plt.show()
# Drop the columns which may not be relevant for predictions

new_df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1)

new_df.head()
# Separate the data frame into input and target variable

Y = new_df['Survived']

X = new_df.drop('Survived', axis=1)
# Function to replace NaN Age values with mean age

def fillna_age(dataframe, mean_age):

    dataframe['Age'] = dataframe.Age.apply(lambda x: x if not pd.isnull(x) else mean_age)

    return dataframe
# Function to convert Sex to numeric values

def convert_sex_numeric(dataframe):

    dataframe['Sex'] = dataframe.Sex.apply(lambda x: 1 if x == 'male' else 0)

    return dataframe
# Replace null 'Age' values with the mean age

mean_age = X['Age'].mean()

X = fillna_age(X, mean_age)

X = convert_sex_numeric(X) 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=1, max_depth=5, min_samples_split=25)

model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

print(accuracy_score(model.predict(X_test), y_test))
# Load test data

test_df = pd.read_csv('../input/test.csv')

test_pid = test_df['PassengerId']

test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Embarked'], axis=1, inplace=True)

test_df.head()
test_df = fillna_age(test_df, mean_age)

test_df = convert_sex_numeric(test_df)
print(test_df.head())

avg_fare = X['Fare'].mean()

test_df['Fare'].fillna(avg_fare, inplace=True)
# Use the trained model to make predictions on the test data

predictions = model.predict(test_df)

predDF = pd.DataFrame(data=predictions, columns=['Survived'])
df1 = pd.DataFrame(test_pid, columns=['PassengerId'])

output = pd.concat([df1, predDF], axis=1)

output.head()
output.to_csv('submission.csv')