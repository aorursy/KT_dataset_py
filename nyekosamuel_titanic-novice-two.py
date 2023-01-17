# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.head()
train_df.info()
train_df.describe()
train_df.shape
train_df.isnull()
train_df.isnull().sum()
missing_val_count = train_df.isnull().sum().sum()

print(f'The number of missing values in the dataset is: {missing_val_count}')
from sklearn.model_selection import train_test_split

X_full = train_df.copy()

X_test_full = test_df.copy()

# Remove rows with missing target 

X_full.dropna(axis=0, subset=['Survived'], inplace=True)

X_full.drop(columns=['Name','Cabin','Ticket'], axis=1, inplace=True)
X_full.head()
encoded_X = pd.get_dummies(X_full[['Sex','Embarked']])

encoded_X
preprocessed_df = pd.concat([X_full, encoded_X], axis=1)

preprocessed_df.head()
y = preprocessed_df.Survived

features_name = preprocessed_df.columns.drop(['Survived','Sex','Embarked'])

print(features_name)

X = preprocessed_df[features_name]
X_train_np, X_valid_np, y_train_np, y_valid_np = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.svm import SVC

# Model without pipeline

# 1. DecisionTreeClassifier

imputer_np = SimpleImputer()

imputed_X_train_np = pd.DataFrame(imputer_np.fit_transform(X_train_np))

imputed_X_valid_np = pd.DataFrame(imputer_np.transform(X_valid_np))

imputed_X_train_np.columns = X_train_np.columns

imputed_X_valid_np.columns = X_valid_np.columns

model_DTC = DecisionTreeClassifier()

model_DTC.fit(imputed_X_train_np, y_train_np)

DTC_pred = model_DTC.predict(imputed_X_valid_np)

mea = mean_absolute_error(y_valid_np, DTC_pred)

score = accuracy_score(y_valid_np, DTC_pred)

print(f'absolute error in thisprediction is: {mea} and accuracy score is:{score}')
# 2. RandomForestRegressor

model_rfg = RandomForestRegressor(n_estimators=100, random_state=1)

model_rfg.fit(imputed_X_train_np, y_train_np)

rfg_pred = model_rfg.predict(imputed_X_valid_np)

mea = mean_absolute_error(y_valid_np, rfg_pred)

#score = accuracy_score(y_valid_np, rfg_pred)

print(f'absolute error in thisprediction is: {mea}')
# 3. SupportVectorMachine

model_svm = SVC(random_state=1)

model_svm.fit(imputed_X_train_np, y_train_np)

svm_pred = model_rfg.predict(imputed_X_valid_np)

mea = mean_absolute_error(y_valid_np, svm_pred)

#score = accuracy_score(y_valid_np, rfg_pred)

print(f'absolute error in thisprediction is: {mea}')
# 2. SupportVectorMachine

model_GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

model_GBC.fit(imputed_X_train_np, y_train_np)

gbc_pred = model_GBC.predict(imputed_X_valid_np)

mea = mean_absolute_error(y_valid_np, gbc_pred)

#score = accuracy_score(y_valid_np, rfg_pred)

print(f'absolute error in thisprediction is: {mea}')

X_test_full.drop(['Cabin','Name','Ticket'], axis=1, inplace=True)

X_test_full.head()
cat_col = ['Sex','Embarked']

encoded_X_test = pd.get_dummies(X_test_full[cat_col])

X_test_combined = pd.concat([X_test_full, encoded_X_test], axis=1)
X_test_combined.head()

#predictiction1 = model_DTC.predict(X_test)
X_test_combined.drop(['Sex','Embarked'],axis=1, inplace=True)
X_test_combined[['Age','Fare']] = X_test_combined[['Age','Fare']].apply(pd.to_numeric)

imputed_X_test = pd.DataFrame(imputer_np.fit_transform(X_test_combined))
imputed_X_test.columns = X_test_combined.columns
imputed_X_test[['PassengerId','Age','Fare']] = imputed_X_test[['PassengerId','Age','Fare']].astype(int)
X_test = imputed_X_test
GBC_pred = model_GBC.predict(X_test)
output = pd.DataFrame({'PassengerId': X_test.PassengerId,

                        'Survived': GBC_pred

                       })

output.to_csv('submission.csv', index=False)