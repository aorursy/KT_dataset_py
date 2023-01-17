# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
# from sklearn.ensemble import RandomForestClassifier



# y = train_data["Survived"]



# features = ["Pclass", "Sex", "SibSp", "Parch"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features])



# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# from sklearn.ensemble import RandomForestClassifier



# y = train_data["Survived"]



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X.Age.fillna(X.Age.mean(), inplace=True) # Fill missing values in Age with mean value

# X_test.Age.fillna(X_test.Age.mean(), inplace=True)



# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features]) 



X.Age.fillna(X.Age.mean(), inplace=True) # Fill missing values in Age with mean value

X_test.Age.fillna(X_test.Age.mean(), inplace=True)



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
# from sklearn.ensemble import RandomForestClassifier



# y = train_data["Survived"]



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X.Age.fillna(X.Age.mean(), inplace=True) # Fill missing values in Age with mean value

# X_test.Age.fillna(X_test.Age.mean(), inplace=True)



# model = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=1)

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# from sklearn.linear_model import LogisticRegression



# y = train_data["Survived"]



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X.Age.fillna(X.Age.mean(), inplace=True) # Fill missing values in Age with mean value

# X_test.Age.fillna(X_test.Age.mean(), inplace=True)



# model = LogisticRegression(max_iter=1000) # Increase max_iter=1000 (from default 100) for convergence

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# from sklearn.neighbors import KNeighborsClassifier



# y = train_data["Survived"]



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X.Age.fillna(X.Age.mean(), inplace=True) # Fill missing values in Age with mean value

# X_test.Age.fillna(X_test.Age.mean(), inplace=True)



# model = KNeighborsClassifier(n_neighbors=5) # Use default settings for all other parameters

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# y = train_data["Survived"]



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X.Age.fillna(X.Age.mean(), inplace=True) # Fill missing values in Age with mean value

# X_test.Age.fillna(X_test.Age.mean(), inplace=True)



# from sklearn.ensemble import RandomForestClassifier

# model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# model_rf.fit(X, y)



# from sklearn.linear_model import LogisticRegression

# model_lr = LogisticRegression(max_iter=1000) # All other parameters default, default max_iter=100

# model_lr.fit(X, y)



# from sklearn.neighbors import KNeighborsClassifier

# model_knn = KNeighborsClassifier(n_neighbors=5) # Use default settings for all other parameters

# model_knn.fit(X, y)



# from sklearn.ensemble import VotingClassifier

# estimators =[('rf', model_rf), ('lr', model_lr), ('knn', model_knn)]

# ensemble = VotingClassifier(estimators, voting='hard') # 'hard' = majority vote

# ensemble.fit(X, y)



# predictions = ensemble.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# from sklearn.ensemble import RandomForestClassifier



# y = train_data["Survived"]



# names_train = train_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_train [['Name_Last', 'Name_Temp']] = names_train['Name'].str.split(',', expand=True)

# names_train [['Name_Title', 'Name_First']] = names_train['Name_Temp'].str.split('.', n=1, expand=True)

# names_train = names_train.drop('Name_Temp', 1)



# names_test = test_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_test [['Name_Last', 'Name_Temp']] = names_test['Name'].str.split(',', n=1, expand=True)

# names_test [['Name_Title', 'Name_First']] = names_test['Name_Temp'].str.split('.', n=1, expand=True)

# names_test = names_test.drop('Name_Temp', 1)



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X['Name_Title'] = names_train['Name_Title'] # Append Name_Title to train_data and test_data

# X_test['Name_Title'] = names_test['Name_Title']



# X['Age'] = X['Age'].fillna(X.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')) # Fill missing Age with mean of Name_Title/Pclass combination e.g. Mr/3

# X['Age'] = X['Age'].fillna(X['Age'].mean()) # Fill any remaining blanks with simple mean of all Ages



# X_test['Age'] = X_test['Age'].fillna(X_test.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean'))

# X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean()) 



# X = X.drop('Name_Title', 1) # Drop Name_Title as only used for filling missing Age, not for modelling

# X_test = X_test.drop('Name_Title', 1)



# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# from sklearn.linear_model import LogisticRegression



# y = train_data["Survived"]



# names_train = train_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_train [['Name_Last', 'Name_Temp']] = names_train['Name'].str.split(',', expand=True)

# names_train [['Name_Title', 'Name_First']] = names_train['Name_Temp'].str.split('.', n=1, expand=True)

# names_train = names_train.drop('Name_Temp', 1)



# names_test = test_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_test [['Name_Last', 'Name_Temp']] = names_test['Name'].str.split(',', n=1, expand=True)

# names_test [['Name_Title', 'Name_First']] = names_test['Name_Temp'].str.split('.', n=1, expand=True)

# names_test = names_test.drop('Name_Temp', 1)



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X['Name_Title'] = names_train['Name_Title'] # Append Name_Title to train_data and test_data

# X_test['Name_Title'] = names_test['Name_Title']



# X['Age'] = X['Age'].fillna(X.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')) # Fill missing Age with mean of Name_Title/Pclass combination e.g. Mr/3

# X['Age'] = X['Age'].fillna(X['Age'].mean()) # Fill any remaining blanks with simple mean of all Ages



# X_test['Age'] = X_test['Age'].fillna(X_test.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean'))

# X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean()) 



# X = X.drop('Name_Title', 1) # Drop Name_Title as only used for filling missing Age, not for modelling

# X_test = X_test.drop('Name_Title', 1)



# model = LogisticRegression(max_iter=1000) # Increase max_iter=1000 (from default 100) for convergence

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# from sklearn.neighbors import KNeighborsClassifier



# y = train_data["Survived"]



# names_train = train_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_train [['Name_Last', 'Name_Temp']] = names_train['Name'].str.split(',', expand=True)

# names_train [['Name_Title', 'Name_First']] = names_train['Name_Temp'].str.split('.', n=1, expand=True)

# names_train = names_train.drop('Name_Temp', 1)



# names_test = test_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_test [['Name_Last', 'Name_Temp']] = names_test['Name'].str.split(',', n=1, expand=True)

# names_test [['Name_Title', 'Name_First']] = names_test['Name_Temp'].str.split('.', n=1, expand=True)

# names_test = names_test.drop('Name_Temp', 1)



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X['Name_Title'] = names_train['Name_Title'] # Append Name_Title to train_data and test_data

# X_test['Name_Title'] = names_test['Name_Title']



# X['Age'] = X['Age'].fillna(X.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')) # Fill missing Age with mean of Name_Title/Pclass combination e.g. Mr/3

# X['Age'] = X['Age'].fillna(X['Age'].mean()) # Fill any remaining blanks with simple mean of all Ages



# X_test['Age'] = X_test['Age'].fillna(X_test.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean'))

# X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean()) 



# X = X.drop('Name_Title', 1) # Drop Name_Title as only used for filling missing Age, not for modelling

# X_test = X_test.drop('Name_Title', 1)



# model = KNeighborsClassifier(n_neighbors=10) # Use default settings for all other parameters

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# y = train_data["Survived"]



# names_train = train_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_train [['Name_Last', 'Name_Temp']] = names_train['Name'].str.split(',', expand=True)

# names_train [['Name_Title', 'Name_First']] = names_train['Name_Temp'].str.split('.', n=1, expand=True)

# names_train = names_train.drop('Name_Temp', 1)



# names_test = test_data[['Name']].copy() # Split Name into Name_First, Name_Title, Name_Last

# names_test [['Name_Last', 'Name_Temp']] = names_test['Name'].str.split(',', n=1, expand=True)

# names_test [['Name_Title', 'Name_First']] = names_test['Name_Temp'].str.split('.', n=1, expand=True)

# names_test = names_test.drop('Name_Temp', 1)



# features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features]) 



# X['Name_Title'] = names_train['Name_Title'] # Append Name_Title to train_data and test_data

# X_test['Name_Title'] = names_test['Name_Title']



# X['Age'] = X['Age'].fillna(X.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean')) # Fill missing Age with mean of Name_Title/Pclass combination e.g. Mr/3

# X['Age'] = X['Age'].fillna(X['Age'].mean()) # Fill any remaining blanks with simple mean of all Ages



# X_test['Age'] = X_test['Age'].fillna(X_test.groupby(['Name_Title', 'Pclass'])['Age'].transform('mean'))

# X_test['Age'] = X_test['Age'].fillna(X_test['Age'].mean()) 



# X = X.drop('Name_Title', 1) # Drop Name_Title as only used for filling missing Age, not for modelling

# X_test = X_test.drop('Name_Title', 1)



# from sklearn.ensemble import RandomForestClassifier

# model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# model_rf.fit(X, y)



# from sklearn.linear_model import LogisticRegression

# model_lr = LogisticRegression(max_iter=1000) # All other parameters default, default max_iter=100

# model_lr.fit(X, y)



# from sklearn.neighbors import KNeighborsClassifier

# model_knn = KNeighborsClassifier(n_neighbors=10) # Use default settings for all other parameters

# model_knn.fit(X, y)



# from sklearn.ensemble import VotingClassifier

# estimators =[('rf', model_rf), ('lr', model_lr), ('knn', model_knn)]

# ensemble = VotingClassifier(estimators, voting='hard') # 'hard' = majority vote

# ensemble.fit(X, y)



# predictions = ensemble.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")