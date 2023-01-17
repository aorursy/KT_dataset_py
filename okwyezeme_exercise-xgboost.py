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

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 15 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)

column_na_count = (X_train.isna().sum())

columns_names_missing_values = [cname for cname in X_train if X_train[cname].isna().any()]

print(f'>>> Columns with missing values: {columns_names_missing_values} \n{column_na_count[column_na_count>0]}')
from xgboost import XGBRegressor



# Define the model

my_model_1 = XGBRegressor(random_state=0) # Your code here



# Fit the model

my_model_1.fit(X_train,y_train) # Your code here



# Check your answer

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
from sklearn.metrics import mean_absolute_error



# Get predictions

predictions_1 = my_model_1.predict(X_valid) # Your code here



# Check your answer

step_1.b.check()
# Lines below will give you a hint or solution code

#step_1.b.hint()

#step_1.b.solution()
# Calculate MAE

mae_1 = mean_absolute_error(y_valid,predictions_1) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_1)



# Check your answer

step_1.c.check()
# Lines below will give you a hint or solution code

#step_1.c.hint()

#step_1.c.solution()
# Define the model

my_model_2 = XGBRegressor(n_estimators=500,random_state=0,learning_rate=0.1) # Your code here



# Fit the model

my_model_2.fit(X_train,y_train,

              eval_set=[(X_valid, y_valid)],

            eval_metric='mae',

            verbose=True,

            early_stopping_rounds=10)



# Get predictions

predictions_2 = my_model_2.predict(X_valid)



# Calculate MAE

mae_2 = mean_absolute_error(y_valid,predictions_2)



# Uncomment to print MAE

print(">>> Mean Absolute Error:" , mae_2)



# Check your answer

step_2.check()
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

import numpy as np

# create and fit the model on the dataset

imputer = SimpleImputer(strategy='most_frequent')

clf = XGBRegressor(n_estimators=252, random_state=0)

# perform imputation

imputed_x_train = pd.DataFrame(data=imputer.fit_transform(X_train),columns=list(X_train))

imputed_x_valid = pd.DataFrame(data=imputer.transform(X_valid),columns=list(X_valid))



clf.fit(imputed_x_train,y_train)

# generate predictions from the validation data

predictions = clf.predict(imputed_x_valid)

# get the threshold iterable by sorting the feature importance

thresholds = np.sort(clf.feature_importances_)

print(f'>>> MAE (without feature selection): {mean_absolute_error(predictions,y_valid)}')



# find the optimum number of features by doing feature selection

num_of_features_list = []

mae_scores = []

for threshold in thresholds:

    selected = SelectFromModel(clf,threshold=threshold,prefit=True)

    selected_train_x = selected.transform(imputed_x_train) # get a reduced version of the train_x

    num_of_features = selected_train_x.shape[1]

    

    # now train a new model with the selected_train_x

    selected_clf = XGBRegressor(n_estimators=252, random_state=0)

    selected_clf.fit(selected_train_x,y_train)

    

    # validate our reduced model

    selected_valid_x = selected.transform(imputed_x_valid)

    predictions = selected_clf.predict(selected_valid_x)

    mae = mean_absolute_error(y_valid,predictions)

    mae_scores.append(mae)

    num_of_features_list.append(num_of_features)

    #print(f'>>> Threshold = {threshold}\t number of features = {num_of_features}\t MAE = {mae}')

    

from matplotlib import pyplot as plt

# let us plot the MAE for the corresponding number of selected features

print(f'>>> Optimal threshold: {thresholds[np.argmin(mae_scores)]}\t Optimal MAE: {np.min(mae_scores)}\t Optimal number of features: {num_of_features_list[np.argmin(mae_scores)]}')

fig, ax = plt.subplots(figsize=(30, 15))

ax.plot(num_of_features_list,mae_scores,color='green', marker='*',linewidth=4, markersize=12)

#ax.plot(num_of_features_imp,mae_imp,color='red', marker='*',linewidth=4, markersize=12,label='Imputed Data')

ax.grid(True)

ax.set_title('MAE vs Number of Features')

ax.set_xlabel('Number of Features')

ax.set_ylabel('MAE')

ax.legend()
# use the regressor from SelectFromModel to make predictions using the threshold that gave us minimum MAE



sfm = SelectFromModel(clf,threshold=0.0008206379134207964,prefit=True)

# generate test output after transforming the test data

imputed_x_test = pd.DataFrame(data=imputer.transform(X_test),columns=list(X_test))

selected_train_x = sfm.transform(imputed_x_train)

selected_test_x = sfm.transform(imputed_x_test)



# create a model for fitting the feature optimized data

selected_model = XGBRegressor(n_estimators=252, random_state=0)

selected_model.fit(selected_train_x,y_train)

predictions = selected_model.predict(selected_test_x)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': predictions})

output.to_csv('submission.csv', index=False)
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Define the model

my_model_3 = XGBRegressor(n_estimators=300,random_state=0,learning_rate=0.01)



# Fit the model

my_model_3.fit(X_train,y_train)# Your code here



# Get predictions

predictions_3 = my_model_3.predict(X_valid)



# Calculate MAE

mae_3 = mean_absolute_error(y_valid,predictions_3)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_3)



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution()