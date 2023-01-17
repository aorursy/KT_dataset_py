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
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
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
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
#import packages
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

def score_dataset(X_train, X_test, y_train, y_test, model):
    model = model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
#------------------------------------------------------------------------------------------------
#import data
#cant find Iowa data so going for melbourne again
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Only numeric predictors. 
data = data.select_dtypes(exclude=['object'])

#split predictors and result
predictors = data
predictors = predictors.drop('Price', axis=1)
target = data.Price

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(predictors, target, train_size=0.7, test_size=0.3, random_state=0)

#------------------------------------------------------------------------------------------------

#dropping columns with missing values
missingCols = [col for col in X_train.columns if X_train[col].isnull().any()]
dropped_X_train = X_train.drop(missingCols, axis=1)
dropped_X_test = X_test.drop(missingCols, axis=1)
errorDroped = score_dataset(dropped_X_train, dropped_X_test, y_train, y_test, RandomForestRegressor())
print('dropping columns give error of: '+str(errorDroped))

#------------------------------------------------------------------------------------------------

#imputing columns with missing values
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
fitImp = imp.fit(X_train)
imputed_X_train = fitImp.transform(X_train)
imputed_X_test = fitImp.transform(X_test)
errorImputed = score_dataset(imputed_X_train, imputed_X_test, y_train, y_test, RandomForestRegressor())
print('Imputing columns give error of: '+str(errorImputed))

#change
print('--- error changed: '+str(errorImputed-errorDroped))

#------------------------------------------------------------------------------------------------

#creating binary columns to trace imputed values
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

for col in missingCols :
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

cols = imputed_X_train_plus.columns

#imputer
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
fitImp = imp.fit(imputed_X_train_plus)
imputed_X_train_plus = fitImp.transform(imputed_X_train_plus)
imputed_X_test_plus = fitImp.transform(imputed_X_test_plus)
errorImputedPlus = score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test, RandomForestRegressor())
print('Imp + Plus columns give error of: '+str(errorImputedPlus))

#change
print('--- error changed: '+str(errorImputedPlus-errorImputed))
test = pd.DataFrame(imputed_X_train_plus, columns=cols)
test.head()