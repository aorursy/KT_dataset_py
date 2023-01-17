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



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 58 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()









# X_train[numeric_cols] = X_train[numeric_cols].apply(

#     lambda x: (x - x.mean()) / (x.std())) # 减均值 除标准差

# # 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值

# X_train[numeric_cols] = X_train[numeric_cols].fillna(0)

# X_valid[numeric_cols] = X_valid[numeric_cols].apply(

#     lambda x: (x - x.mean()) / (x.std())) # 减均值 除标准差

# # 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值

# X_valid[numeric_cols] = X_valid[numeric_cols].fillna(0)

# X_test[numeric_cols] = X_test[numeric_cols].apply(

#     lambda x: (x - x.mean()) / (x.std())) # 减均值 除标准差

# # 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值

# X_test[numeric_cols] = X_test[numeric_cols].fillna(0)







# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_train.shape, X_valid.shape,X_test.shape
# X_train.isna().sum()
# X_train
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error,mean_squared_error

my_model = XGBRegressor(n_estimators=(1400), learning_rate=0.04) # Your code here

my_model.fit(X_train, y_train)

# my_model.fit(X_train, y_train, 

#              early_stopping_rounds=5, 

#              eval_set=[(X_valid, y_valid)],

#              verbose=False)

preds = my_model.predict(X_valid)

MAE = mean_absolute_error(y_valid, preds)

print('MAE:', MAE)
my_model.score(X_valid,y_valid)
preds_test = my_model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)

my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))

cv_scores = cross_val_score(my_pipeline, X_valid, y_valid, 

                            cv=5,

                            scoring='accuracy')



print("Cross-validation accuracy: %f" % cv_scores.mean())
# FOR 1

for i in range(12,15):# 7891011

#     print(i)

    my_model_1 = XGBRegressor(n_estimators=(100*i), learning_rate=0.04) # Your code here

    my_model_1.fit(X_train, y_train)

    preds = my_model_1.predict(X_valid)

    MAE = mean_absolute_error(y_valid, preds)

    print('MAE:', MAE)



# Check your answer
# FOR 2

for i in range(2,6):

#     print((i+1))

    my_model_2 = XGBRegressor(n_estimators=(1000), learning_rate=(0.01*(i+1))) # Your code here

    my_model_2.fit(X_train, y_train)

    preds = my_model_2.predict(X_valid)

    MAE = mean_absolute_error(y_valid, preds)

    print('MAE:', MAE)



# Check your answer
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=(900), learning_rate=0.03) # Your code here

my_model.fit(X_train, y_train)

preds = my_model.predict(X_valid)

MAE = mean_absolute_error(y_valid, preds)

print('MAE:', MAE)
preds_test = my_model.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission9.csv', index=False)
# from xgboost import XGBRegressor

# my_model_1 = XGBRegressor(n_estimators=900, learning_rate=0.03) # Your code here

# # Define the model

# my_model_1 = XGBRegressor(random_state=0) # Your code here



# # Fit the model

# my_model_1.fit(X_train,y_train)# Your code here

# preds = my_model_1.predict(X_valid)

# MAE = mean_absolute_error(y_valid, preds)

# print('MAE:', MAE)

# # Check your answer

# Lines below will give you a hint or solution code

#step_1.a.hint()

step_1.a.solution()
from sklearn.metrics import mean_absolute_error,mean_squared_error





# Get predictions

predictions_1 = my_model_1.predict(X_valid) # Your code here



# Check your answer

step_1.b.check()
# Lines below will give you a hint or solution code

#step_1.b.hint()

#step_1.b.solution()
# Calculate MAE

mae_1 = mean_absolute_error(predictions_1, y_valid) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_1)



# Check your answer

step_1.c.check()
# Lines below will give you a hint or solution code

#step_1.c.hint()

step_1.c.solution()
my_model_2 = XGBRegressor(n_estimators=(1000), learning_rate=(3)) # Your code here

my_model_2.fit(X_train,y_train) # Your code here

predictions_2 = my_model_2.predict(X_valid) # Your code here

mae_2 = mean_absolute_error(predictions_2,y_valid) # Your code here

print("Mean Absolute Error:" , mae_2)
# Define the model

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=2) # Your code here



# Fit the model

my_model_2.fit(X_train,y_train) # Your code here



# Get predictions

predictions_2 = my_model_2.predict(X_valid) # Your code here



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2,y_valid) # Your code here

# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)



# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Define the model

my_model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)



# Fit the model

# my_model_3.fit(X_train,y_train) # Your code here

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

step_3.solution()
preds_test = my_model_2.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission9.csv', index=False)
