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
Col_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

print(Col_with_missing)
Col_with_missing = [col for col in test_data.columns if test_data[col].isnull().any()]

print(Col_with_missing)
# Get list of categorical variables

s = (train_data.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
feature_name=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

X=train_data[feature_name]

y=train_data["Survived"]

X_test=test_data[feature_name]
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
from sklearn.impute import SimpleImputer

my_imputer=SimpleImputer(strategy="most_frequent")

imputed_X_train= pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_test=pd.DataFrame(my_imputer.transform(X_test))

imputed_X_valid=pd.DataFrame(my_imputer.transform(X_valid))

imputed_X_train.index = X_train.index

imputed_X_valid.index = X_valid.index

imputed_X_test.index = X_test.index

imputed_X_train.columns=X_train.columns

imputed_X_valid.columns=X_valid.columns

imputed_X_test.columns=X_test.columns
Col_with_missing_2 = [col for col in imputed_X_test.columns if imputed_X_test[col].isnull().any()]

print(Col_with_missing_2)
# Feature Generation

New_feature_train = imputed_X_train['Sex'] + "_" + imputed_X_train['Embarked']

New_feature_valid = imputed_X_valid['Sex'] + "_" + imputed_X_valid['Embarked']

New_feature_test = imputed_X_test['Sex'] + "_" + imputed_X_test['Embarked']
imputed_X_train["Sex_Embarked"]=New_feature_train

imputed_X_valid["Sex_Embarked"]=New_feature_valid

imputed_X_test["Sex_Embarked"]=New_feature_test
imputed_X_test.head()
Cat_cols=['Sex','Embarked','Sex_Embarked']
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train[Cat_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(imputed_X_valid[Cat_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(imputed_X_test[Cat_cols]))

OH_cols_train.index = imputed_X_train.index

OH_cols_valid.index = imputed_X_valid.index

OH_cols_test.index = imputed_X_test.index

num_X_train = imputed_X_train.drop(Cat_cols, axis =1)

num_X_valid = imputed_X_valid.drop(Cat_cols, axis =1)

num_X_test = imputed_X_test.drop(Cat_cols, axis =1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
OH_X_train = OH_X_train.apply(pd.to_numeric)

OH_X_valid = OH_X_valid.apply(pd.to_numeric)

OH_X_test = OH_X_test.apply(pd.to_numeric)

OH_X_train=OH_X_train.rename(columns={0:"Sex1", 1:"Sex2"})

OH_X_train=OH_X_train.rename(columns={2:"C", 3:"Q",4:"S"})

OH_X_valid=OH_X_valid.rename(columns={0:"Sex1", 1:"Sex2"})

OH_X_valid=OH_X_valid.rename(columns={2:"C", 3:"Q",4:"S"})

OH_X_test=OH_X_test.rename(columns={0:"Sex1", 1:"Sex2"})

OH_X_test=OH_X_test.rename(columns={2:"C", 3:"Q",4:"S"})
OH_X_test.head(10)
from xgboost import XGBClassifier

from sklearn import metrics

my_model = XGBClassifier(n_estimators=1000, learning_rate=0.001)

my_model.fit(OH_X_train, y_train, early_stopping_rounds=50, 

             eval_set=[(OH_X_valid, y_valid)], verbose=False)

my_model.fit(OH_X_train, y_train)

y_pred5 = my_model.predict(OH_X_valid)

print("Accuracy:",metrics.accuracy_score(y_valid, y_pred5))
predictions2 = my_model.predict(OH_X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions2})

output.to_csv('my_submission_02_06.csv', index=False)

print("Your submission was successfully saved!")