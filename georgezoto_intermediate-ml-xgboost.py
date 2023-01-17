!lscpu
!free -m
# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex6 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)

print('X_train_full.shape: ', X_train_full.shape)

# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()

print('X_train.shape: ', X_train.shape)



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)



print('Before align')

print('\n--- X_train.columns ---\n', str(list(X_train.columns)), '\n--- X_valid.columns ---\n', str(list(X_valid.columns)), '\n--- X_test.columns ---\n', str(list(X_test.columns)))

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)

print('After align')

print('\n--- X_train.columns ---\n', str(list(X_train.columns)), '\n--- X_valid.columns ---\n', str(list(X_valid.columns)), '\n--- X_test.columns ---\n', str(list(X_test.columns)))
from xgboost import XGBRegressor



# Define the model

my_model_1 = XGBRegressor(random_state=0)



# Fit the model

my_model_1.fit(X_train, y_train)



# Check your answer

step_1.a.check()
print(my_model_1)
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
from sklearn.metrics import mean_absolute_error



# Get predictions

predictions_1 = my_model_1.predict(X_valid)



# Check your answer

step_1.b.check()
# Lines below will give you a hint or solution code

#step_1.b.hint()

#step_1.b.solution()
# Calculate MAE

mae_1 = mean_absolute_error(predictions_1, y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_1)



# Check your answer

step_1.c.check()
# Lines below will give you a hint or solution code

#step_1.c.hint()

#step_1.c.solution()
import time
n_estimators=3000

learning_rate=0.01

n_jobs=-1

early_stopping_rounds=10
start = time.perf_counter()



# Define the model

my_model_2 = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=n_jobs)



# Fit the model

my_model_2.fit(X_train, y_train)



#my_model_2.fit(X_train, y_train, 

#             early_stopping_rounds=early_stopping_rounds, 

#             eval_set=[(X_valid, y_valid)], 

#             verbose=False)



# Get predictions

predictions_2 = my_model_2.predict(X_valid)



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2, y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)



end = time.perf_counter()



print("n_estimators: {}, learning_rate: {}, n_jobs: {}, early_stopping_rounds: {}, average MAE score:{}, time (sec):{:.2}"\

    .format(n_estimators, learning_rate, n_jobs, early_stopping_rounds, mae_2.mean(), (end-start)))

    

# Check your answer

#step_2.check()
preds_test = my_model_2.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
print(my_model_1)
print(my_model_2)
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Define the model

my_model_3 = XGBRegressor(n_estimators=10, learning_rate=0.1)



# Fit the model

my_model_3.fit(X_train, y_train)



# Get predictions

predictions_3 = my_model_3.predict(X_valid)



# Calculate MAE

mae_3 = mean_absolute_error(predictions_3, y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_3)



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution()