# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from learntools.core import *



from sklearn.model_selection import RandomizedSearchCV

import numpy as np



from sklearn.impute import SimpleImputer







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)



#Option 1: Select certain columns 



#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#X = home_data[features]



# Split into validation and training data

#train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


#Option 2: Select only numeric columns without nulls 



def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)





# Load data

home_data_target = home_data.SalePrice

home_data_predictors = home_data.drop(['SalePrice','Id'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors.

home_data_numeric_predictors = home_data_predictors.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(home_data_numeric_predictors, home_data_target,

                                                    train_size=0.7, test_size=0.3, random_state=42)

cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]









# Method 1  **************************

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_test = X_test.drop(cols_with_missing, axis=1)

print("Mean Absolute Error from dropping columns with Missing Values:")

print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))



# Method 2 ************************

my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(X_train)

imputed_X_test = my_imputer.transform(X_test)

print("Mean Absolute Error from Imputation:")

print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))



print('\n\n');



# Method 3 **************************

imputed_X_train_plus = X_train.copy()

imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())

for col in cols_with_missing:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

my_imputer = SimpleImputer()

imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)

imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")

print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

print('\n');

    

train_X = reduced_X_train

val_X = reduced_X_test



train_y =  y_train

val_y =  y_test


##Head

print(train_X.head())



##Numeric columns list

print(list(reduced_X_train.columns.values))

print('\n\n\n');



## option 1



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



print(random_grid)



## option 2

'''



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]

#max_depth.append(None)



# Minimum number of samples required to split a node

min_samples_split = [5, 10]

# Minimum number of samples required at each leaf node

#min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

#bootstrap = [True, False]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split}



print(random_grid)



'''
# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores



##option 1

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



##option 2

#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='neg_mean_squared_error')





# Fit the random search model

rf_random.fit(train_X, train_y)
print(rf_random.best_params_)
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    print(' ')

    return accuracy



base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)

base_model.fit(train_X, train_y)

base_accuracy = evaluate(base_model,   val_X, val_y  )





   

best_random = rf_random.best_estimator_

random_accuracy = evaluate(best_random,  val_X, val_y  )





print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data =  rf_random.best_estimator_



# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features



#Option 1: Basic Column Selection

#featuresTest = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#test_X = test_data[featuresTest]



#Option 2: Only Numeric Columns

featuresTest = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

test_X = test_data[featuresTest]

test_X = test_X.fillna(test_X.mean())



test_preds = rf_model_on_full_data.predict(test_X)







print(test_preds)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.





output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

print('CSV GENERATED !!!')