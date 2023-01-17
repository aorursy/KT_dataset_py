# Set up code checking

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

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

# X_train = pd.get_dummies(X_train)

# X_valid = pd.get_dummies(X_valid)

# X_test = pd.get_dummies(X_test)

# X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

# X_train, X_test = X_train.align(X_test, join='left', axis=1)

X_train.head()
from xgboost import XGBRegressor





# Define the model

my_model_1 = XGBRegressor(random_state=0)



# Fit the model

my_model_1.fit( X_train, y_train )



# Check your answer

step_1.a.check()
#%%RM_IF(PROD)%

from sklearn.utils.validation import check_is_fitted

my_model_1 = XGBRegressor(random_state=0)

step_1.a.assert_check_failed()
#%%RM_IF(PROD)%

my_model_1 = XGBRegressor(random_state=0)

my_model_1.fit(X_train, y_train)

step_1.a.assert_check_passed()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
from sklearn.metrics import mean_absolute_error



# Get predictions

predictions_1 = my_model_1.predict(X_valid)



# Check your answer

step_1.b.check()
#%%RM_IF(PROD)%

predictions_1 = my_model_1.predict(X_valid)

step_1.b.assert_check_passed()
# Lines below will give you a hint or solution code

#step_1.b.hint()

#step_1.b.solution()
# Calculate MAE

mae_1 = mean_absolute_error(predictions_1,y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_1)



# Check your answer

step_1.c.check()
#%%RM_IF(PROD)%

mae_1 = mean_absolute_error(predictions_1, y_valid)

print("Mean Absolute Error:" , mae_1)

step_1.c.assert_check_passed()
# Lines below will give you a hint or solution code

#step_1.c.hint()

#step_1.c.solution()
# Define the model

my_model_2 = XGBRegressor( n_estimators = 1000, learning_rate=0.05 )



# Fit the model

my_model_2.fit( X_train,y_train )



# Get predictions

predictions_2 = my_model_2.predict( X_valid )



# Calculate MAE

mae_2 = mean_absolute_error( predictions_2, y_valid )



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)



# Check your answer

step_2.check()
#%%RM_IF(PROD)%

my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model_2.fit(X_train, y_train)

predictions_2 = my_model_2.predict(X_valid)

mae_2 = mean_absolute_error(predictions_2, y_valid)

print("Mean Absolute Error:" , mae_2)



step_2.assert_check_passed()
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Define the model

my_model_3 = XGBRegressor( n_estimators=1, random_state=0)



# Fit the model

my_model_3.fit(X_train, y_train)



# Get predictions

predictions_3 = my_model_3.predict(X_valid)



# Calculate MAE

mae_3 = mean_absolute_error(predictions_3,y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_3)



# Check your answer

step_3.check()
#%%RM_IF(PROD)%

my_model_3 = XGBRegressor(n_estimators=1)

my_model_3.fit(X_train, y_train)

predictions_3 = my_model_3.predict(X_valid)

mae_3 = mean_absolute_error(predictions_3, y_valid)

print("Mean Absolute Error:" , mae_3)



step_3.assert_check_passed()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution()

from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer



numerical_transformer = SimpleImputer( strategy = "constant" )



categorical_transofmer = Pipeline(

                            steps=[

                            ('imputer',  SimpleImputer( strategy = "constant" )),

                            ('onehot',   OneHotEncoder( handle_unknown = "ignore" ))                                

                        ])



preprocessor = ColumnTransformer(

                transformers=[

                    ('num', numerical_transformer, numeric_cols ),

                    ('cat', categorical_transofmer, low_cardinality_cols ),

                ])



myModel = XGBRegressor( n_estimator = 1000, learning_rate=0.05, random_state = 0 )



myPipeline = Pipeline(

                steps=[

                    ('preprocessor', preprocessor),

                    ('model', myModel),

                ])



myPipeline.fit( X_train, y_train )



predictions = myPipeline.predict(X_test)



output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': predictions})

output.to_csv('submission.csv', index=False)