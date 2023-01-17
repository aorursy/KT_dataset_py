import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



raw_train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

example = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
from sklearn.model_selection import train_test_split



# Select subset of predictors

cols_to_use = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

X = raw_train[cols_to_use]



# Select target

y = raw_train.Survived



# Separate data into training and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
X_train.head()
X_train.describe()
features = ['Pclass','Age','SibSp','Parch','Fare']

pd.plotting.scatter_matrix(X_train[features], figsize=(10,10))

plt.show()
def PrintColsWithMissing(data):

    cols = [col for col in data.columns

                     if data[col].isnull().any()]

    return cols

print(PrintColsWithMissing(X_train))

print(PrintColsWithMissing(X_valid))

print(PrintColsWithMissing(test))
from sklearn.impute import SimpleImputer

# Imputation

my_imputer = SimpleImputer()

f = ['Age', 'Fare']

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train[f]))

X_train['Age'] = imputed_X_train[0].to_numpy()

X_train['Fare'] = imputed_X_train[1].to_numpy()



imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid[f]))

X_valid['Age'] = imputed_X_valid[0].to_numpy()

X_valid['Fare'] = imputed_X_valid[1].to_numpy()



imputed_X_test = pd.DataFrame(my_imputer.transform(test[f]))

test['Age'] = imputed_X_test[0].to_numpy()

test['Fare'] = imputed_X_test[1].to_numpy()
print(PrintColsWithMissing(X_train))

print(PrintColsWithMissing(X_valid))

print(PrintColsWithMissing(test))
X_train.head()
from sklearn.preprocessing import OneHotEncoder



f = ['Sex']



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[f]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[f]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test[f]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index

OH_cols_test.index = test.index



# Add one-hot encoded columns to numerical features

X_train = pd.concat([X_train, OH_cols_train], axis=1)

X_valid = pd.concat([X_valid, OH_cols_valid], axis=1)

test = pd.concat([test, OH_cols_test], axis=1)
X_train = X_train.rename(columns={0: "sex1", 1: "sex2"})

X_valid = X_valid.rename(columns={0: "sex1", 1: "sex2"})

test = test.rename(columns={0: "sex1", 1: "sex2"})
X_train.head()
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'sex1', 'sex2']

pd.plotting.scatter_matrix(X_train[features], figsize=(10,10))

plt.show()
from xgboost import XGBClassifier



my_model = XGBClassifier()

my_model.fit(X_train[features], y_train)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



predictions = my_model.predict(X_valid[features])

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

print("Mean Absolute Error: " + str(accuracy_score(predictions, y_valid)))
comp_predictions = my_model.predict(test[features])

to_submit = pd.concat([test.PassengerId, pd.Series(comp_predictions)], axis=1)

to_submit = to_submit.rename(columns={0:'Survived'})

to_submit.to_csv('out.csv', columns = ('PassengerId', 'Survived'), index = False)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(X_valid[features], y_valid)

eli5.show_weights(perm, feature_names = features)
features2 = ['Pclass', 'Age', 'Fare', 'sex1']

my_model2 = XGBClassifier()

my_model2.fit(X_train[features2], y_train)



predictions = my_model2.predict(X_valid[features2])

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

print("Mean Absolute Error: " + str(accuracy_score(predictions, y_valid)))



comp_predictions = my_model2.predict(test[features2])

to_submit = pd.concat([test.PassengerId, pd.Series(comp_predictions)], axis=1)

to_submit = to_submit.rename(columns={0:'Survived'})

to_submit.to_csv('out2.csv', columns = ('PassengerId', 'Survived'), index = False)
features3 = ['Age', 'SibSp', 'Parch', 'Fare']

bad_model = XGBClassifier()

bad_model.fit(X_train[features3], y_train)



predictions = bad_model.predict(X_valid[features3])

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

print("Mean Absolute Error: " + str(accuracy_score(predictions, y_valid)))