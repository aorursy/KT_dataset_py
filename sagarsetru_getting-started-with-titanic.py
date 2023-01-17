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
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

women = train_data.loc[train_data.Sex == 'female']['Survived']
# print(women)
rate_women = sum(women)/len(women)
# print('percent of women who survived: ',rate_women)

men = train_data.loc[train_data.Sex == 'male']['Survived']
rate_men = sum(men)/len(men)
# print('percent of men who survived: ',rate_men)

# determine which cols have missing data
print(train_data)
train_data.isna().any()
print(' ')

# impute age values
train_data['Age'].fillna(train_data['Age'].mean(), inplace = True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace = True)

# impute embarked values
train_data['Embarked'].fillna(train_data['Embarked'].mode(), inplace = True)
test_data['Embarked'].fillna(test_data['Embarked'].mode(), inplace = True)

# impute fare
train_data['Fare'].fillna(train_data['Fare'].mean(), inplace = True)
test_data['Fare'].fillna(test_data['Fare'].mean(), inplace = True)

# use zero for unknown cabin
train_data.Cabin = train_data.Cabin.notnull().astype('int')
test_data.Cabin = test_data.Cabin.notnull().astype('int')
# train_data['Cabin'].fillna(0, inplace = True)
# test_data['Cabin'].fillna(0, inplace = True)
print('train data cabin after imputation')
print(train_data.Cabin)

print('test data cabin after imputation')
print(test_data.Cabin)

print(train_data)
# # train_data.Cabin.loc[~train_data.Cabin.isnull()] = 1  # not nan
# train_data.Cabin.loc[train_data.Cabin.isnull()] = 0  # nan
# # test_data.Cabin.loc[~test_data.Cabin.isnull()] = 1  # not nan
# test_data.Cabin.loc[test_data.Cabin.isnull()] = 0  # nan

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# get output target variables
y = train_data['Survived']

# get list of features
features_1 = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age', 'Cabin']

# get train and test features
X_train = pd.get_dummies(train_data[features_1])
X_test = pd.get_dummies(test_data[features_1])

X_train, X_val, y_train, y_val = train_test_split(X_train, y, random_state = 0)

# build random forest model
model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# build decision tree model
model_dt = DecisionTreeClassifier(random_state=0, max_depth=5)

# build support vector machine models
model_svm_rbf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale'))
model_svm_lin = make_pipeline(StandardScaler(), SVC(kernel='linear'))
model_svm_poly = make_pipeline(StandardScaler(), SVC(kernel='poly'))

# fit the models to the training data
model_rf.fit(X_train,y_train)

model_dt.fit(X_train,y_train)

model_svm_rbf.fit(X_train,y_train)
model_svm_lin.fit(X_train,y_train)
model_svm_poly.fit(X_train,y_train)

# test on validation data
pred_val_rf = model_rf.predict(X_val)
print('validation error random forest: ',mean_absolute_error(y_val,pred_val_rf))

pred_val_dt = model_dt.predict(X_val)
print('validation error decision tree: ',mean_absolute_error(y_val,pred_val_dt))

pred_val_svm_rbf = model_svm_rbf.predict(X_val)
print('validation error svm with radial basis kernel: ',mean_absolute_error(y_val,pred_val_svm_rbf))

pred_val_svm_lin = model_svm_lin.predict(X_val)
print('validation error svm with linear kernel: ',mean_absolute_error(y_val,pred_val_svm_lin))

pred_val_svm_poly = model_svm_poly.predict(X_val)
print('validation error svm with polynomial kernel: ',mean_absolute_error(y_val,pred_val_svm_poly))


# make prediction
predictions_rf = model_rf.predict(X_test)

predictions_dt = model_dt.predict(X_test)

predictions_svm_rbf = model_svm_rbf.predict(X_test)

predictions_svm_lin = model_svm_lin.predict(X_test)

predictions_svm_poly = model_svm_poly.predict(X_test)

# boolean for whether to show output
showOutputDFs = 0

# make data frames for submission
# for random forest
output_rf = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions_rf})
output_rf.to_csv('submission_rf.csv', index=False)

# for decision tree
output_dt = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions_dt})
output_dt.to_csv('submission_dt.csv', index=False)

# for svm with various kernels
output_svm_rbf = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions_svm_rbf})
output_svm_rbf.to_csv('submission_svm_rbf.csv', index=False)

output_svm_lin = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions_svm_lin})
output_svm_lin.to_csv('submission_svm_lin.csv', index=False)

output_svm_poly = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions_svm_poly})
output_svm_poly.to_csv('submission_svm_poly.csv', index=False)

if showOutputDFs == 1:
    print('output data frame for random forest')
    print(output_rf)

    print('output data frame for decision tree')
    print(output_dt)
    
    print('output data from for svm with rbf kernel')
    print(output_svm_rbf)
    
    print('output data from for svm with lin kernel')
    print(output_svm_lin)
    
    print('output data from for svm with poly kernel')
    print(output_svm_poly)
#...

print('Your submissions successfully saved!')


