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
from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv', index_col='sl_no')





# Remove rows with missing target

X_full.dropna(axis=0, subset=['salary'], inplace=True)



import matplotlib.pyplot as plt # Visualize Salary (Target) distribution



plt.close()

X_full.salary.hist(bins = 30)

plt.show()
# From the histogram, it appears that the salary is not normally distributed

# The Shapiro-Wilk test can be used as a test of normality

import scipy.stats as stat



stat.shapiro(X_full.salary)
# Exploratory Phase (Visualizations)

import matplotlib.pyplot as plt



print(X_full.info())

print("\n")

print(X_full.head())



plt.clf(); plt.cla(); plt.close();

X_full["gender"].hist()

plt.show()



X_full["degree_t"].hist()

plt.show()



X_full[X_full.gender == "M"]["salary"].hist()

X_full[X_full.gender == "F"]["salary"].hist()

plt.legend(["M","F"])

plt.show()



X_full[X_full.degree_t == "Sci&Tech"]["salary"].hist(alpha = 0.6)

X_full[X_full.degree_t == "Comm&Mgmt"]["salary"].hist(alpha = 0.6)

X_full[X_full.degree_t == "Others"]["salary"].hist(alpha = 0.6)

plt.legend(["Sci&Tech","Comm&Mgmt","Others"])

plt.show()


# Separate target from predictors

y = X_full.salary

X_full.drop(['salary'], axis=1, inplace=True)



# Break off validation set from training data

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

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error





# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy = "most_frequent")



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy = "most_frequent")),

    ('onehot', OneHotEncoder(handle_unknown = "ignore"))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators = 100, random_state = 0)



# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

print('MAE:', score)



# Fitting using XGBRegressor

from xgboost import XGBRegressor



xgb_model = XGBRegressor(n_estimators = 550, learning_rate = 0.05)



xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model',xgb_model)

                             ])



xgb_pipeline.fit(X_train,y_train)



xgb_preds = xgb_pipeline.predict(X_valid)



# Evaluate the model

xgb_score = mean_absolute_error(y_valid, xgb_preds)

print('MAE:', score)

# Preprocessing of test data, fit model

preds_test = my_pipeline.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'Salary': preds_test})

output.to_csv('submission.csv', index=False)


