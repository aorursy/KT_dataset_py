#Ready data 

import pandas as pd

train_file_path='../input/train.csv'

train_data=pd.read_csv(train_file_path)

train_data.head()

type(train_data)#pandas.core.frame.DataFrame

train_data.shape#(1460, 81) include SalePrice

test_file_path='../input/test.csv'

test_data=pd.read_csv(test_file_path)

test_data.head()

type(test_data)#pandas.core.frame.DataFrame

test_data.shape#(1460, 80) exclude SalePrice
#check the missing value

missing_val_count_by_column = (train_data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column>0].index)

#Drop Columns with Missing Values

cols_with_missing = [col for col in train_data.columns 

                                 if train_data[col].isnull().any()]

len(cols_with_missing)#19

reduced_train_data = train_data.drop(cols_with_missing, axis=1)

reduced_test_data = test_data.drop(cols_with_missing, axis=1)



reduced_train_data.shape#(1460, 62) Dropped 19 Columns

reduced_test_data.shape#(1459, 61) Dropped 19 Columns



target = reduced_train_data.SalePrice

predictors = reduced_train_data.drop(['SalePrice'], axis=1)

predictors.head()

num_predictors = predictors.select_dtypes(exclude=['object'])

num_predictors.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(num_predictors, 

                                                    target,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)



def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor(n_estimators=75, max_depth=10,oob_score=True)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)



print(score_dataset(X_train, X_test, y_train, y_test))
#Imputation

from sklearn.impute import SimpleImputer



# For the sake of keeping the example simple, we'll use only numeric predictors. 

numeric_predictors = train_data.select_dtypes(exclude=['object'])



# make copy to avoid changing original data (when Imputing)

new_data = numeric_predictors.copy()



# make new columns indicating what will be imputed

#cols_with_missing = (col for col in new_data.columns 

#                                 if new_data[col].isnull().any())

#for col in cols_with_missing:

#    new_data[col + '_was_missing'] = new_data[col].isnull()

#new_data.head()



target = new_data.SalePrice

predictors = new_data.drop(['SalePrice'], axis=1)

predictors.head()



my_imputer = SimpleImputer()

imputed_predictors=my_imputer.fit_transform(predictors)



imputed_X_train, imputed_X_test, y_train, y_test = train_test_split(imputed_predictors, 

                                                    target,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)

print("Mean Absolute Error from Imputation:")

print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))