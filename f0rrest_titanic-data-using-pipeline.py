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
X_full = pd.read_csv('../input/titanic/train.csv')

X_test_full = pd.read_csv('../input/titanic/test.csv')



X_full.head()
print("The mean age for class 3 passengers is:\t", X_full.Age.loc[X_full.Pclass == 3].median())

print("The mean age for class 2 passengers is:\t", X_full.Age.loc[X_full.Pclass == 2].median())

print("The mean age for class 1 passengers is:\t", X_full.Age.loc[X_full.Pclass == 1].median())
print("The mean fare for class 3 passengers is:\t", X_full.Fare.loc[X_full.Pclass == 3].median())

print("The mean fare for class 2 passengers is:\t", X_full.Fare.loc[X_full.Pclass == 2].median())

print("The mean fare for class 1 passengers is:\t", X_full.Fare.loc[X_full.Pclass == 1].median())
def get_age(data, i):

    if data['Age'][i] == 0:

        if data['Pclass'][i] == 3:

            data['Age'][i] = 24.0

        elif data['Pclass'][i] == 2:

            data['Age'][i] = 29.0

        else:

            data['Age'][i] = 37.0

    return
def get_fare(data, i):

    if data['Age'][i] == 0:

        if data['Pclass'][i] == 3:

            data['Fare'][i] = 8.05

        elif data['Pclass'][i] == 2:

            data['Fare'][i] = 14.25

        else:

            data['Fare'][i] = 60.287499999999994

    return
# Fill in NaN values with 0

X_full['Age'].fillna(0, inplace=True)

X_test_full['Age'].fillna(0, inplace=True)



# Fill in 0 values with mean ages based on Pclass in train_data

for i in range(len(X_full.Age)):

    get_age(X_full, i)



for i in range(len(X_test_full.Age)):

    get_age(X_test_full, i)
# Fill in NaN values with 0

X_full['Fare'].fillna(0, inplace=True)

X_test_full['Fare'].fillna(0, inplace=True)



# Fill in 0 values with mean ages based on Pclass in train_data

for i in range(len(X_full.Fare)):

    get_age(X_full, i)



for i in range(len(X_test_full.Fare)):

    get_age(X_test_full, i)
from sklearn.model_selection import train_test_split



y = X_full.Survived

X_full.drop(['Survived'], axis=1, inplace=True)



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

# my_cols = categorical_cols + numerical_cols

# X_train = X_train_full[my_cols].copy()

# X_valid = X_valid_full[my_cols].copy()

# X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



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
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# # Preprocessing of training data, fit model 

my_pipeline.fit(X_full, y)



# # Preprocessing of validation data, get predictions

# preds = my_pipeline.predict(X_valid)



from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X_full, y,

                              cv=5,

                              scoring='average_precision')



print("MAE scores:\n", scores)
test_preds = my_pipeline.predict(X_test_full)



print(test_preds)
output = pd.DataFrame({'PassengerId': X_test_full.PassengerId, 'Survived': test_preds})

output.to_csv('my_submission.csv', index=False)



print("your submission was successfully saved!")