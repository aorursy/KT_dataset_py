import numpy as np 

import pandas as pd

import math

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print("Proceed.")
#Loading train data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()



# age_missing_train = []



# for x in train_data['Age']:

#     if math.isnan(x) == True:

#         age_missing_train.append(True) # True

#     else:

#         age_missing_train.append(False) # False

        

# train_data['Age_Missing'] = age_missing_train



mean_age_train = train_data['Age'].mean()



train_data['Age'] = train_data['Age'].fillna(mean_age_train)

train_data['Embarked'] = train_data['Embarked'].fillna('C')



train_data['Family']= train_data['SibSp'] + train_data['Parch']



#train_data.isnull().sum()
train_data.head()

#Loading test data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()



# age_missing_test = []



# for x in test_data['Age']:

#     if math.isnan(x) == True:

#         age_missing_test.append(True) # True

#     else:

#         age_missing_test.append(False) # False

        

# test_data['Age_Missing'] = age_missing_test



mean_age_test = test_data['Age'].mean()

mean_fare_test = test_data['Fare'].mean()



test_data['Age'] = test_data['Age'].fillna(mean_age_test)

test_data['Fare'] = test_data['Fare'].fillna(mean_fare_test)



test_data['Family']= test_data['SibSp'] + test_data['Parch']



#test_data.isnull().sum()
train_y = train_data["Survived"]



features = ["Pclass", "Sex" ,"Age", "SibSp", "Parch", "Fare", "Embarked"] 

# Pclass=passenger class, SibSp=sibling/spouse, Parch=parents/children



train_X = pd.get_dummies(train_data[features])

val_X = pd.get_dummies(test_data[features])
model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_split=2, random_state=0)

model.fit(train_X, train_y)

predictions = model.predict(val_X)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)



print(output.head())