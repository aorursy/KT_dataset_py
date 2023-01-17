# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex2 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split

#LodeStar has addded the option below

pd.set_option('display.max_columns', 100)



# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



#print(X_full.head())



#LodeStar: Check shape of original X_full and X_test_full datasets 

print("X_full Shape :", X_full.shape) 

print("X_test_full Shape :", X_test_full.shape) 



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



#LodeStar: Check shape of X_full and X_test_full AFTER removing rows with missing target 

print("X_full Shape (after row removal for missing target) :", X_full.shape) 

print("X_test_full Shape (no row removal done here) :", X_test_full.shape) 



# To keep things simple, we'll use only numerical predictors

X = X_full.select_dtypes(exclude=['object'])

X_test = X_test_full.select_dtypes(exclude=['object'])



#LodeStar: Check shape of X_full and X_test_full using numerical predictors  

print("X Shape (after selecting for numerical predictors ONLY) :", X.shape) 

print("X_test Shape (after selecting for numerical predictors ONLY) :", X_test.shape) 



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)



#LodeStar: Checking shapes of X_train, X_valid, y_train and y_valid after breaking off (TRAIN+VALID SIZE = FULL SIZE) 

print("X_train Shape :", X_train.shape, "X_valid Shape :", X_valid.shape) 

print("y_train Shape :", y_train.shape, "y_valid Shape :", y_valid.shape) 
X_full.head()
X.head() 

#LodeStar: One can also do X.tail() 
X_test.head()

#LodeStar: One can also do X_test.tail()  
X_train.head() 

#LodeStar: One can also do X_train.tail() 
X_valid.head()

#LodeStar: One can also do X_valid.tail() 
# Shape of training data (num_rows, num_columns)

print(X_train.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])



#LodeStar requested this printout 

print(X_train.columns)
# Fill in the line below: How many rows are in the training data?

num_rows = 1168



# Fill in the line below: How many columns in the training data

# have missing values?

num_cols_with_missing = 3



# Fill in the line below: How many missing entries are contained in 

# all of the training data?

tot_missing = 276 #(212+6+58)



# Check your answers

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
#step_1.b.hint()
#step_1.b.solution()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
#Fill in the line below: get names of columns with missing values. #Put your code here 

cols_missing_data = [col for col in X_train.columns 

                        if X_train[col].isnull().any()] 



#LodeStar: Let's see which columns had missing values

print(len(cols_missing_data)) 

print(cols_missing_data)



#Fill in the lines below: drop columns in training and validation data 

reduced_X_train = X_train.drop(cols_missing_data, axis=1) 

reduced_X_valid = X_valid.drop(cols_missing_data, axis=1)



#Check your answers 

step_2.check()
#After dropping columns with missing data in training dataset 

reduced_X_train.shape

#LodeStar: One can also use the head()/tail() method 
#After dropping columns with missing data in validation dataset 

reduced_X_valid.shape

#LodeStar: One can also use the head()/tail() method
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
print("MAE (Drop columns with missing values):")

print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
from sklearn.impute import SimpleImputer



print("Before SimpleImputer is let loose", X_train.shape, X_valid.shape) 



# Fill in the lines below: imputation. Put your code here 

my_imputer = SimpleImputer() 



imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid)) 



# Fill in the lines below: imputation removed column names; put them back

imputed_X_train.columns = X_train.columns 

imputed_X_valid.columns = X_valid.columns



print("After SimpleImputer is done", imputed_X_train.shape, imputed_X_valid.shape) 



# Check your answers

step_3.a.check()
# Lines below will give you a hint or solution code

#step_3.a.hint()

#step_3.a.solution()
print("MAE (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
#step_3.b.hint()
step_3.b.solution()
# Lines below will give you a hint or solution code

#step_4.a.hint()

#step_4.a.solution()
#Approach 1: LodeStar will try the "median imputation strategy" with the Imputer as discussed above in hints

app1_imputer = SimpleImputer(strategy='median')



#LodeStar: Checking shapes 

print("Before Median Imputation Strategy is let loose", X_train.shape, X_valid.shape) 



# Preprocessed training and validation features

final_X_train = pd.DataFrame(app1_imputer.fit_transform(X_train)) 

final_X_valid = pd.DataFrame(app1_imputer.transform(X_valid)) 



#Imputation removes column names. So putting them back 

final_X_train.columns = X_train.columns 

final_X_valid.columns = X_valid.columns 



#LodeStar verifying that the training and validation datasets have the same number of rows 

print("final_X_train Shape =", final_X_train.shape, "final_y_train Shape =", y_train.shape)

print("final_X_valid Shape =", final_X_valid.shape, "final_y_valid Shape =", y_valid.shape) 



#LodeStar verifying that the preprocessed DataFrames have the same number of columns 

# For final_X_train and final_X_valid 

print(final_X_train.columns.shape, final_X_valid.columns.shape) 

print(final_X_train.index.shape, final_X_valid.index.shape) #DataFrame object has no attribute 'rows', use 'index' instead ! 



#LodeStar verifying that the preprocessed X_train has no missing values. 

missing_val_count_by_column = final_X_train.isnull().sum() 

print(missing_val_count_by_column[missing_val_count_by_column > 0]) 



#LodeStar verifying that the preprocessed X_valid has no missing values. 

missing_val_X_valid = final_X_valid.isnull().sum() 

print(missing_val_X_valid[missing_val_X_valid > 0])



# Check your answers

step_4.a.check()
X_train.head()
final_X_train.head()
X_valid.head()
final_X_valid.head()
# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(final_X_train, y_train) 



# Get validation predictions and MAE

preds_valid = model.predict(final_X_valid) 



print("Shape of preds_valid :", preds_valid.shape)

print("MAE (Your approach): strategy='median'")

print(mean_absolute_error(y_valid, preds_valid))
#Approach 2: LodeStar will try the "extended imputation strategy" with the Imputer now

app2_imputer = SimpleImputer() 



#Find the columns with missing data 

cols_missing_data = [col for col in X_train.columns 

                        if X_train[col].isnull().any()]



print(cols_missing_data, len(cols_missing_data)) 



#Making copies of training and validation datasets before altering their structures 

X_train_copy = X_train.copy() 

X_valid_copy = X_valid.copy() 



#Creating extra columns containing info on which data were imputed 

for col in cols_missing_data: 

    X_train_copy[col + '_was_missing'] = X_train_copy[col].isnull()

    X_valid_copy[col + '_was_missing'] = X_valid_copy[col].isnull() 

    

imputed_X_train_copy = pd.DataFrame(app2_imputer.fit_transform(X_train_copy)) 

imputed_X_valid_copy = pd.DataFrame(app2_imputer.transform(X_valid_copy)) 



#Imputation removes column names, so put them back 

imputed_X_train_copy.columns = X_train_copy.columns 

imputed_X_valid_copy.columns = X_valid_copy.columns 



print("Shapes for X_train and X_valid :", X_train.shape, X_valid.shape)

print("Shapes for imputed X_train and X_valid :", imputed_X_train_copy.shape, imputed_X_valid_copy.shape)



missing_val_count_by_column = imputed_X_train_copy.isnull().sum() 

print(missing_val_count_by_column[missing_val_count_by_column > 0]) 



missing_val_count_by_column = imputed_X_valid_copy.isnull().sum() 

print(missing_val_count_by_column[missing_val_count_by_column > 0]) 

#LodeStar; Checking the original training dataset 

X_train.head()
#LodeStar: Checking the Imputed dataset #VVI: Notice how the "Id" column is changed completely ! 

imputed_X_train_copy.head()
#LodeStar: Checking the original validation dataset 

X_valid.head()
#LodeStar: Checking the Imputed dataset #VVI: Notice how the "Id" column is changed completely !

imputed_X_valid_copy.head()
# Define and fit model

model = RandomForestRegressor(n_estimators=100, random_state=0)

model.fit(imputed_X_train_copy, y_train)



# Get validation predictions and MAE

preds_valid = model.predict(imputed_X_valid_copy)

print("Shape of preds_valid :", preds_valid.shape)

print("MAE (Your approach): Extended Imputation Strategy")

print(mean_absolute_error(y_valid, preds_valid))
#LodeStar: Checking X_test dataset 

# >> print("Before Median Imputation Strategy acts on X_test, its shape :", X_test.shape)

print("Before Extended Imputation Strategy acts on X_test, its shape :", X_test.shape)



#Making a copy before changing the structure 

X_test_copy = X_test.copy()



print(cols_missing_data)



#Creating extra columns containing info on which data were imputed  

for col in cols_missing_data: 

    X_test_copy[col + '_was_missing'] = X_test_copy[col].isnull()



# Fill in the line below: preprocess test data 

# >> final_X_test = pd.DataFrame(app1_imputer.transform(X_test)) 

final_X_test = pd.DataFrame(app2_imputer.transform(X_test_copy)) 



#LodeStar: I also tried to see the impact of "final_X_test = pd.DataFrame(app1_imputer.fit_transform(X_test))", but it was not as good as just "transform". 

#LodeStar: This is probably because the fit results from a bigger dataset i.e. X_train are more reliable than X_test. 



#Imputation removes column names. So putting them back 

final_X_test.columns = X_test_copy.columns 



print("After Extended Imputation Strategy acts on X_test, its shape :", final_X_test.shape)



# Fill in the line below: get test predictions

preds_test = model.predict(final_X_test) 



step_4.b.check()
X_test.head()
final_X_test.head()
# Lines below will give you a hint or solution code

#step_4.b.hint()

#step_4.b.solution()
# Save test predictions to file

#LodeStar: This cell DOES need change in the assignment of "Id". Id should be assigned according to the original X_test file, NOT according to final_X_test.

#output = pd.DataFrame({'Id': final_X_test.index,

                       #'SalePrice': preds_test})

output = pd.DataFrame({'Id': X_test.index, 

                      'SalePrice': preds_test}) 

output.to_csv('submission.csv', index=False)