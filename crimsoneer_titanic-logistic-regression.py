import pandas as pd

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LogisticRegression



test = pd.read_csv("test.csv")

train = pd.read_csv("train.csv")



train.info()

test.info()

test.isnull().sum()

train["Name_Length"] = train["Name"].str.len()

test["Name_Length"] = test["Name"].str.len()

sex_dummy = pd.get_dummies(train["Sex"])

embarked_dummy = pd.get_dummies(train["Embarked"])

class_dummy = pd.get_dummies(train["Pclass"])



train = pd.concat([train, sex_dummy, embarked_dummy, class_dummy], axis=1)



sex_dummy = pd.get_dummies(test["Sex"])

embarked_dummy = pd.get_dummies(test["Embarked"])

class_dummy = pd.get_dummies(test["Pclass"])



test = pd.concat([test, sex_dummy, embarked_dummy,  class_dummy], axis=1)



train.info()
test.info()

test.isnull().sum()
from sklearn.impute import SimpleImputer



import numpy as np



imr = SimpleImputer(missing_values= np.nan, strategy='mean')

imr = imr.fit(test[['Age']])

test['Age'] = imr.transform(test[['Age']]).ravel()



imr = imr.fit(test[['Fare']])

test['Fare'] = imr.transform(test[['Fare']]).ravel()





test.info()

test.isnull().sum()



imr = imr.fit(train[['Age']])

train['Age'] = imr.transform(train[['Age']]).ravel()



train.info()

train.isnull().sum()



test.info()

features = ["Age", "Fare", "Name_Length", "female", "male", "C", "Q", "S", 1, 2, 3]



log_model = LogisticRegression(max_iter=1000)

log_model.fit(train[features], train["Survived"])



features = ["Age", "Fare", "Name_Length", "female", "male", "C", "Q", "S", 1, 2, 3]



test["predictions"] = log_model.predict(test[features])
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test["predictions"]})



output.to_csv('logistic_submission.csv', index=False)

print("Your submission was successfully saved!")