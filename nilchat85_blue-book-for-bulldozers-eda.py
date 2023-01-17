# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import data
bulldozer_train_path = "../input/train/Train.csv"
# df_raw = pd.read_csv(f"{training_data_path}Train.csv",
#                      low_memory= False, 
#                      parse_dates=["saledate"])

df_bulldozer_train = pd.read_csv(bulldozer_train_path, low_memory=False)
# Find the size of the training data
df_bulldozer_train.shape
# Check the statistical summary of the numeric data
df_bulldozer_train.describe()
# Check the statistical summary of the non-numeric data
df_bulldozer_train.describe(include='O')
# Check the number of columns
df_bulldozer_train.columns
# Check the top 5 rows
df_bulldozer_train.head()
# Writing a function to display all the columns and rows within the data set
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)
def add_datepart(df, fldname):
    fld = df[fldname]
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    for n in ('Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 'Is_quarter_end', 
              'Is_quarter_start', 'Is_year_end', 'Is_year_start'):
        df[targ_pre+n] = getattr(fld.dt, n.lower())
    df[targ_pre+'Elapsed'] = (fld - fld.min()).dt.days
    df.drop(fldname, axis=1, inplace=True)
# # Drop the missing values
# df_bulldozer_cleaned = df_bulldozer.dropna(axis=1)
# df_bulldozer_cleaned.head()
# Selecting The Prediction Target
y = df_bulldozer_train.SalePrice
# Determine the non numeric features
bulldozer_features = df_bulldozer_train.drop(['SalePrice'], axis= 1).select_dtypes(exclude='O').columns

X = df_bulldozer_train[bulldozer_features]
X.describe()
# Check for missing values in X
missing_columns = [col for col in X.columns if X[col].isnull().any()]
missing_columns 
# Handling missing values in X
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)
# Split the data into train and validation set
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.33, random_state = 42)
# def get_mae(max_leaf_nodes, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val):
#     model = DecisionTreeRegressor(random_state=1, max_leaf_nodes= max_leaf_nodes)
#     model.fit(X_train, y_train)
#     return mean_absolute_error(y_val, model.predict(X_val))

# nodes = [5,10,25,50, 75, 100, 150, 200, 250, 275, 300, 350, 400]
# for node in nodes:
#     print(get_mae(node))
# # Building a decision tree model
# model_dt = DecisionTreeRegressor(random_state=1, max_leaf_nodes=150)

# # Fit model
# model_dt.fit(X_train, y_train)

# # Predict
# y_pred = model_dt.predict(X_val)

# # Model Evaluation
# mean_absolute_error(y_val, y_pred)
# Define model
model = RandomForestRegressor(random_state= 1)

# Fit model
model.fit(X_imputed, y)
# Read validation data
bulldozer_valid_path = "../input/Valid.csv"
df_bulldozer_valid = pd.read_csv(bulldozer_valid_path, low_memory=False)
# Modify X_val as per the test features
X_val = df_bulldozer_valid[bulldozer_features]
X_val_imputed = imputer.transform(X_val)
# Predict using validation set
y_pred_val = model.predict(X_val_imputed)
# Read validation solution data
bulldozer_valid_soln_path = "../input/ValidSolution.csv"
df_bulldozer_valid_soln = pd.read_csv(bulldozer_valid_soln_path, low_memory=False)
y_val = df_bulldozer_valid_soln.SalePrice
# Evaludate the model
mean_absolute_error(y_val, y_pred_val)
# vectorized root mean square log error
def rmsle(y_pred, y_test):
    assert len(y_pred) == len(y_test)
    return np.sqrt(np.mean(np.power(np.log1p(y_pred)-np.log1p(y_test), 2)))

rmsle(y_pred_val, y_val)
# Read test data
bulldozer_test_path = "../input/Test.csv"
df_bulldozer_test = pd.read_csv(bulldozer_test_path, low_memory=False)

# Update the features in the test data
X_test = df_bulldozer_test[bulldozer_features]

# Impute test data
X_test_imputed = imputer.transform(X_test)

# Predict on test data
y_test_pred = model.predict(X_test_imputed)
# Create output

output = pd.DataFrame({'SalesID': df_bulldozer_test.SalesID,
                       'SalePrice': y_test_pred})
output.to_csv('submission.csv', index= False)