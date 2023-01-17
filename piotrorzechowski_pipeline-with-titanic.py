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



train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# women = train_data.loc[train_data.Sex == 'female']["Survived"]

# rate_women = sum(women)/len(women)



# print("% of women who survived:", rate_women)



# men = train_data.loc[train_data.Sex == 'male']["Survived"]

# rate_men = sum(men) / len(men)

# print(predictions)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

prediction_test = model.predict(X_test)

predictions_train = model.predict(X)

print(accuracy_score(y, predictions_train))



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': prediction_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier



# Read the data

X_full = pd.read_csv("/kaggle/input/titanic/train.csv")

X_test_full = pd.read_csv("/kaggle/input/titanic/test.csv")



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['Survived'], inplace=True)

y = X_full.Survived

X_full.drop(['Survived'], axis=1, inplace=True)

X_full.drop(['Name'], axis=1, inplace=True)

X_full.drop(['PassengerId'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 20 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()

X_train.head()







# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

# model = RandomForestClassifier(n_estimators=1000, random_state=0)

model = XGBClassifier(learning_rate=0.02, n_estimators=7500,

                   max_depth= 3, min_child_weight= 1, 

                   colsample_bytree= 0.6, gamma= 0.0, 

                   reg_alpha= 0.001, subsample= 0.8)

# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model 

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_test)

preds_valid = clf.predict(X_valid)

print('done')

X_full.head()
print('ACC:', accuracy_score(y_valid, preds_valid))





output = pd.DataFrame({'PassengerId': X_test_full.PassengerId, 'Survived': preds})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
