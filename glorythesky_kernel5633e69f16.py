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
from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd 

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Import training and testing data

train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
#Which columns have missing values?

display(train_data.isnull().sum().sort_values(ascending=False))

display(test_data.isnull().sum().sort_values(ascending=False))
#Create new Cabin variable

train_data['Cabin_new'] = train_data['Cabin'].str[0]

test_data['Cabin_new'] = train_data['Cabin'].str[0]



#Create title variable

train_data['Title']=0

train_data['Title']=train_data.Name.str.extract('([A-Za-z]+)\.')



test_data['Title']=0

test_data['Title']=test_data.Name.str.extract('([A-Za-z]+)\.')
y = train_data["Survived"]

X = train_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "Cabin_new", "Title"]]

X_test = test_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked", "Cabin_new", "Title"]]



cat_cols = [cname for cname in X.columns if

                    X[cname].nunique() < 10 and 

                    X[cname].dtype == "object"]



# Select numerical columns

num_cols = [cname for cname in X.columns if 

                X[cname].dtype in ['int64', 'float64']]
#Preprocessing numerical data

num_transformer = SimpleImputer()



#Preprocessing for categorical data

cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



#Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, num_cols),

        ('cat', cat_transformer, cat_cols)])



#Model

#model = RandomForestClassifier(random_state=126)

model = XGBRegressor(random_state=126)



#Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocess', preprocessor), 

                              ('model', model)])



#For random forest

#param_grid = {

#    'model__n_estimators': [20, 30, 40],

#    'model__max_depth': [10, 15, 20]}



param_grid = {

    'model__n_estimators': [20, 30, 40],

    'model__learning_rate': [0.09, 0.1, 0.2]}





search = GridSearchCV(my_pipeline, param_grid, n_jobs=-1, verbose=10, cv=5)

search.fit(X, y)

print("Best parameter (CV score=%0.3f):" % search.best_score_)

print(search.best_params_)
predictions = search.predict(X_test)

predictions = [0 if x <= 0.5 else 1 for x in predictions]



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")