import pandas as pd

# Load data
melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns 
                                 if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
import pandas as pd
import numpy as np

# Load data
train = pd.read_csv('../input/iowa-house-prices/train.csv') 
test = pd.read_csv('../input/iowa-house-prices/test.csv')

train_target = train.SalePrice
train_predictors = train.drop(['SalePrice'], axis = 1)

# Combining both train and test sets
full = pd.concat([train_predictors,test])

# For simplicity's sake, full has been subsetted to retain only numeric values
full_numeric_predictors = full.select_dtypes(exclude=['object'])
# Missing values per column
missing_values_count = full.isnull().sum()

# Percentage of total data cells missing
total_cells = np.product(full.shape)
total_missing = missing_values_count.sum()

(total_missing/total_cells) * 100
from sklearn.ensemble import RandomForestRegressor

xtrain = train_predictors.select_dtypes(exclude=['object'])
ytrain = train_target

xtest = test.select_dtypes(exclude=['object'])

def price_dataset(xtrain, xtest, ytrain):
    model = RandomForestRegressor()
    model.fit(xtrain,ytrain)
    return model.predict(xtest)
# Obtain predictions by removing dropped columns
cols_with_missing = [col for col in full_numeric_predictors.columns
                                if full_numeric_predictors[col].isnull().any()]
reduced_xtrain = xtrain.drop(cols_with_missing, axis=1)
reduced_xtest = xtest.drop(cols_with_missing, axis=1)
ytest = price_dataset(reduced_xtrain, reduced_xtest, ytrain)

# save our predictions to csv
solution = pd.DataFrame({"id":test.Id, "SalePrice":ytest})
solution.to_csv("droppedcolumnsprediction.csv", index = False)

# Obtain predictions using imputation
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_xtrain = my_imputer.fit_transform(xtrain)
imputed_xtest = my_imputer.fit_transform(xtest)
ytest = price_dataset(imputed_xtrain, imputed_xtest, ytrain)

# save our predictions to csv
solution = pd.DataFrame({"id":test.Id, "SalePrice":ytest})
solution.to_csv("imputedcolumnsprediction.csv", index = False)
# Obtain predictions using an extension to imputation
imputed_xtrainplus = xtrain.copy()
imputed_xtestplus = xtest.copy()

cols_with_missing = [col for col in full_numeric_predictors.columns
                                if full_numeric_predictors[col].isnull().any()]

for col in cols_with_missing:
    imputed_xtrainplus[col + 'was_missing'] = imputed_xtrainplus[col].isnull()
    imputed_xtestplus[col + 'was_missing'] = imputed_xtestplus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_xtrainplus = my_imputer.fit_transform(imputed_xtrainplus)
imputed_xtestplus = my_imputer.fit_transform(imputed_xtestplus)
ytest = price_dataset(imputed_xtrainplus, imputed_xtestplus, ytrain)

# save our predictions to csv
solution = pd.DataFrame({"id":test.Id, "SalePrice":ytest})
solution.to_csv("extimputedcolumnsprediction.csv", index = False)
