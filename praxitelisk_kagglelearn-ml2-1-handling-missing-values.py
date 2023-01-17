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
'''
My turn to write down some code and tackle the problem is NA values :D
'''
#read Melborne_housing_FULL csv dataset
import pandas as pd

full_data = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")
#descriptive statistics
full_data.describe()
#remove the rows with NAs method before passing by the training (solution #1)
full_data = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")

full_data = full_data.dropna(axis=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target = full_data.Price
melb_predictors = full_data.drop(['Price'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
melb_numeric_predictors = full_data.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(X_train, X_test, y_train, y_test))
#solution #2, fill NAs with Imputer class

from sklearn.preprocessing import Imputer

#read csv
full_data = pd.read_csv("../input/melbourne-housing-market/Melbourne_housing_FULL.csv")

# For the sake of keeping the example simple, we'll use only numeric predictors. 
full_data = full_data.select_dtypes(exclude=['object'])
columns = full_data.columns

# create an Imputer class to fill NAs with mean from the respective columns
my_imputer = Imputer()
full_data = my_imputer.fit_transform(melb_numeric_predictors)

#because Imputer class has turned the main dataframe from pandas.Dataframe to numpy.array, it must be turned back to a dataframe one.
full_data =  pd.DataFrame(full_data, columns=columns)


melb_target = full_data.Price
melb_predictors = full_data.drop(['Price'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(melb_predictors, 
                                                    melb_target,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=0)


print("Mean Absolute Error from Imputation:")
print(score_dataset(X_train, X_test, y_train, y_test))