# installing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import mode
import string

#loading data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

print("train data describe")
print(train_data.describe())

print("test data describe")
print(test_data.describe())
#check variables names
print("train data variable names")
print(train_data.columns)

print("test data variable names")
print(test_data.columns)
#print 40 rows of train data
train_data.head(10)
#store the dependant variable: Survived
survived_data = train_data.Survived
# head command returns the top few lines of data.
print(survived_data.head())
#store ID variable
id_data = train_data.PassengerId
print(id_data.head())
id_data2 = test_data.PassengerId
print(id_data2.head())
#check variables with missing values
print(train_data.isnull().sum())
print(test_data.isnull().sum())
#first BRUTAL try drop all variables with missing data in both train and test datasets

cols_with_missing = [col for col in train_data.columns 
                                 if train_data[col].isnull().any()]
cols_with_missing2 = [col for col in test_data.columns 
                                 if test_data[col].isnull().any()]
reduced_train_data = train_data.drop(cols_with_missing, axis=1)
reduced_test_data = test_data.drop(cols_with_missing2, axis=1)
#check
print(reduced_train_data.isnull().sum())
print(reduced_test_data.isnull().sum())
X_train = reduced_train_data
X_test = reduced_test_data
X_train.head()
X_test.head()
X_train = X_train.drop(['Name', 'Ticket','Fare','Survived'], axis=1)
X_test = X_test.drop(['Name','Ticket','Embarked'], axis=1)
#create predictors
#train_predictors = ['PassengerId', 'Pclass', 'Sex', 'SibSp',
#       'Parch']
#X_train = reduced_train_data[train_predictors]

#test_predictors = ['PassengerId', 'Pclass', 'Name', 'Sex', 'SibSp',
 #      'Parch', 'Ticket', 'Embarked']
#X_test = reduced_test_data[test_predictors]

print(X_train.head())
print(X_test.head())

X_train.dtypes
X_test.dtypes
one_hot_encoded_train_predictors = pd.get_dummies(X_train)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)

final_train, final_test = one_hot_encoded_train_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
print(final_train.isnull().sum())
print(final_test.isnull().sum())
#from sklearn.tree import DecisionTreeRegressor

# Define model
#titanic_model = DecisionTreeRegressor()

# Fit model
#titanic_model.fit(final_train, survived_data)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

titanic_model = RandomForestRegressor()
titanic_model.fit(final_train, survived_data)

#titanic_preds = titanic_model.predict(final_train)
#print(mean_absolute_error(survived_data, titanic_preds))
print("Making predictions for the following 5 passengers:")
print(final_train.head())
print("The predictions are")
print(titanic_model.predict(final_train.head()))
from sklearn.metrics import mean_absolute_error

predicted_survived = titanic_model.predict(final_test)
#mean_absolute_error(survived_data, predicted_survived)
print(predicted_survived)
predicted_survived = np.around(predicted_survived,0)
print(predicted_survived)
pina_submission = pd.DataFrame({'PassengerId': id_data2, 'Survived': predicted_survived})
# you could use any filename. We choose submission here
pina_submission['Survived'] = pina_submission['Survived'].astype(np.int64)
pina_submission.to_csv('submission.csv', index=False)
pina_submission.describe()
pina_submission.head(20)

