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
from sklearn.svm import SVC

classifier = SVC(kernel='rbf', random_state = 1);

y = train_data["Survived"]



features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]

X = train_data[features]

X_test = test_data[features]

X["Sex"] = np.where(X["Sex"]=="male", 1, 0)

X = X.fillna(X.median())

classifier.fit(X,y);



X_test["Sex"] = np.where(X_test["Sex"]=="male", 1, 0)

X_test = X_test.fillna(X_test.median())

Y_pred = classifier.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission_svm.csv', index=False)
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex","Age", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X = X.fillna(X.median())

X_test = pd.get_dummies(test_data[features])

X_test = X_test.fillna(X_test.median())

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_rf.csv', index=False)

print("Your submission was successfully saved!")
y = train_data["Survived"]

train_data['Age_Class'] = train_data['Age']*train_data['Pclass']

train_data['Family_Size'] = train_data['SibSp']+train_data['Parch']

features = ["Age_Class", "Sex", "Family_Size", "Fare"]



#X = pd.get_dummies(train_data[features])

X = train_data[features]

X["Sex"] = np.where(X["Sex"]=="male", 1, 0)

X = X.fillna(X.median())



test_data['Age_Class'] = test_data['Age']*test_data['Pclass']

test_data['Family_Size'] = test_data['SibSp']+test_data['Parch']

#X_test = pd.get_dummies(test_data[features])

X_test = test_data[features]



X_test["Sex"] = np.where(X_test["Sex"]=="male", 1, 0)

X_test = X_test.fillna(X_test.median())

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_rf3.csv', index=False)

print("Your submission was successfully saved!")

y = train_data["Survived"]

train_data['Age_Class'] = train_data['Age']*train_data['Pclass']

train_data['Family_Size'] = train_data['SibSp']+train_data['Parch']

features = ["Age","Pclass","Age_Class", "Sex", "Family_Size", "Fare"]



#X = pd.get_dummies(train_data[features])

X = train_data[features]

X["Sex"] = np.where(X["Sex"]=="male", 1, 0)

X = X.fillna(X.median())



test_data['Age_Class'] = test_data['Age']*test_data['Pclass']

test_data['Family_Size'] = test_data['SibSp']+test_data['Parch']

#X_test = pd.get_dummies(test_data[features])

X_test = test_data[features]



X_test["Sex"] = np.where(X_test["Sex"]=="male", 1, 0)

X_test = X_test.fillna(X_test.median())

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_rf4.csv', index=False)

print("Your submission was successfully saved!")

y = train_data["Survived"]

train_data['Age_Class'] = train_data['Age']*train_data['Pclass']

train_data['Family_Size'] = train_data['SibSp']+train_data['Parch']

features = ["Age","Pclass","Age_Class", "Sex", "Family_Size", "Fare", "Embarked"]



#X = pd.get_dummies(train_data[features])

X = train_data[features]

X["Sex"] = np.where(X["Sex"]=="male", 1, 0)

#X["Embarked"] = np.where(X["Embarked"]=="C", 0, X["Embarked"])



conditions = [

    (X["Embarked"] == 'C'),

    (X["Embarked"] == 'Q'),

    (X["Embarked"] == 'S')]

choices = [0,1,2]



X["Embarked"] = np.select(conditions, choices)

X = X.fillna(X.median())



test_data['Age_Class'] = test_data['Age']*test_data['Pclass']

test_data['Family_Size'] = test_data['SibSp']+test_data['Parch']

#X_test = pd.get_dummies(test_data[features])

X_test = test_data[features]





X_test["Sex"] = np.where(X_test["Sex"]=="male", 1, 0)



conditions = [

    (X_test["Embarked"] == 'C'),

    (X_test["Embarked"] == 'Q'),

    (X_test["Embarked"] == 'S')]

choices = [0,1,2]



X_test["Embarked"] = np.select(conditions, choices)

X_test = X_test.fillna(X_test.median())

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_rf5.csv', index=False)

print("Your submission was successfully saved!")
y = train_data["Survived"]

train_data['Age_Class'] = train_data['Age']*train_data['Pclass']

train_data['Family_Size'] = train_data['SibSp']+train_data['Parch']

features = ["Age","Pclass","Age_Class", "Sex", 'SibSp', 'Parch', 'Family_Size', "Fare", "Embarked"]



#X = pd.get_dummies(train_data[features])

X = train_data[features]

X["Sex"] = np.where(X["Sex"]=="male", 1, 0)

#X["Embarked"] = np.where(X["Embarked"]=="C", 0, X["Embarked"])



conditions = [

    (X["Embarked"] == 'C'),

    (X["Embarked"] == 'Q'),

    (X["Embarked"] == 'S')]

choices = [0,1,2]



X["Embarked"] = np.select(conditions, choices)

X = X.fillna(X.median())



test_data['Age_Class'] = test_data['Age']*test_data['Pclass']

test_data['Family_Size'] = test_data['SibSp']+test_data['Parch']

#X_test = pd.get_dummies(test_data[features])

X_test = test_data[features]





X_test["Sex"] = np.where(X_test["Sex"]=="male", 1, 0)



conditions = [

    (X_test["Embarked"] == 'C'),

    (X_test["Embarked"] == 'Q'),

    (X_test["Embarked"] == 'S')]

choices = [0,1,2]



X_test["Embarked"] = np.select(conditions, choices)

X_test = X_test.fillna(X_test.median())

model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_rf6.csv', index=False)

print("Your submission was successfully saved!")
y = train_data["Survived"]

train_data['Age_Class'] = train_data['Age']*train_data['Pclass']

train_data['Family_Size'] = train_data['SibSp']+train_data['Parch']

features = ["Age","Pclass","Age_Class", "Sex", 'SibSp', 'Parch', "Fare", "Embarked"]



#X = pd.get_dummies(train_data[features])

X = train_data[features]

X["Sex"] = np.where(X["Sex"]=="male", 1, 0)

#X["Embarked"] = np.where(X["Embarked"]=="C", 0, X["Embarked"])



conditions = [

    (X["Embarked"] == 'C'),

    (X["Embarked"] == 'Q'),

    (X["Embarked"] == 'S')]

choices = [0,1,2]



X["Embarked"] = np.select(conditions, choices)

X = X.fillna(X.median())



test_data['Age_Class'] = test_data['Age']*test_data['Pclass']

test_data['Family_Size'] = test_data['SibSp']+test_data['Parch']

#X_test = pd.get_dummies(test_data[features])

X_test = test_data[features]





X_test["Sex"] = np.where(X_test["Sex"]=="male", 1, 0)



conditions = [

    (X_test["Embarked"] == 'C'),

    (X_test["Embarked"] == 'Q'),

    (X_test["Embarked"] == 'S')]

choices = [0,1,2]



X_test["Embarked"] = np.select(conditions, choices)

X_test = X_test.fillna(X_test.median())

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_rf7.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

train_data['Age_Class'] = train_data['Age']*train_data['Pclass']

train_data['Family_Size'] = train_data['SibSp']+train_data['Parch']

train_data["Sex"] = np.where(train_data["Sex"]=="male", 1, 0)

#X["Embarked"] = np.where(X["Embarked"]=="C", 0, X["Embarked"])



conditions = [

    (train_data["Embarked"] == 'C'),

    (train_data["Embarked"] == 'Q'),

    (train_data["Embarked"] == 'S')]

choices = [0,1,2]



train_data["Embarked"] = np.select(conditions, choices)



y = train_data["Survived"]

features = ["Age_Class", "Sex", 'SibSp', 'Parch', "Fare", "Embarked"]



X = train_data[features]



X = X.fillna(X.median())





X_train,X_valid,y_train,y_valid=train_test_split(X,y, test_size=0.2, random_state=1)



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

print(accuracy_score(y_valid,y_pred))
features = ["Age","Pclass","Age_Class", "Sex", "Fare", "Embarked","Family_Size"]





X = train_data[features]



X = X.fillna(X.median())





X_train,X_valid,y_train,y_valid=train_test_split(X,y, test_size=0.2, random_state=1)



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_valid)

print(accuracy_score(y_valid,y_pred))




#X = pd.get_dummies(train_data[features])



test_data['Age_Class'] = test_data['Age']*test_data['Pclass']

test_data['Family_Size'] = test_data['SibSp']+test_data['Parch']

#X_test = pd.get_dummies(test_data[features])

X_test = test_data[features]





X_test["Sex"] = np.where(X_test["Sex"]=="male", 1, 0)



conditions = [

    (X_test["Embarked"] == 'C'),

    (X_test["Embarked"] == 'Q'),

    (X_test["Embarked"] == 'S')]

choices = [0,1,2]



X_test["Embarked"] = np.select(conditions, choices)

X_test = X_test.fillna(X_test.median())



predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_rf8.csv', index=False)

print("Your submission was successfully saved!")