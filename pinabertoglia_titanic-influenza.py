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
#check type of variable
train_data.dtypes
#check type of variable
test_data.dtypes
#Drop PassengerId, Name, Cabin and Ticket from test and train AND Survived from train

X_train = train_data.drop(['PassengerId','Name','Cabin','Ticket','Survived'], axis=1)
X_test = test_data.drop(['PassengerId','Name','Cabin','Ticket'], axis=1)
#Check
X_train.head()
#Check
X_test.head()
one_hot_encoded_train_data = pd.get_dummies(X_train)
one_hot_encoded_test_data = pd.get_dummies(X_test)

final_train, final_test = one_hot_encoded_train_data.align(one_hot_encoded_test_data,
                                                                    join='left', 
                                                                    axis=1)
print(final_train.isnull().sum())
print(final_test.isnull().sum())
#Trying imputation

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values_train = my_imputer.fit_transform(final_train)
data_with_imputed_values_test = my_imputer.fit_transform(final_test)

#check
print(data_with_imputed_values_train)
#check
print(data_with_imputed_values_test)
#Let's model

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

titanic_model = RandomForestRegressor()
titanic_model.fit(data_with_imputed_values_train, survived_data)

#titanic_preds = titanic_model.predict(final_train)
#print(mean_absolute_error(survived_data, titanic_preds))
#from sklearn.metrics import mean_absolute_error

predicted_survived = titanic_model.predict(data_with_imputed_values_test)
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

