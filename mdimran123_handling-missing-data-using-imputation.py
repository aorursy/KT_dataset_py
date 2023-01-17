import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
lowa_data=pd.read_csv('../input/train.csv')
target=lowa_data.SalePrice
#selecting the targer variable
lowa_predictors=lowa_data.drop(['SalePrice'],axis=1)
#keeping only the numeric columns and dropping any catgorical data columns.
lowa_num_pred=lowa_predictors.select_dtypes(exclude=['object'])
lowa_num_pred.columns
X_train,X_test,y_train,y_test=train_test_split(lowa_num_pred,target,train_size=0.7,test_size=0.3,random_state=1)
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
mis_col=[col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train=X_train.drop(mis_col,axis=1)
reduced_X_test=X_test.drop(mis_col,axis=1)
print("mean abs error from dropping columns with Missing values")
print(score_dataset(reduced_X_train,reduced_X_test,y_train,y_test))
from sklearn.preprocessing import Imputer
my_imputer=Imputer()
imputed_X_train=my_imputer.fit_transform(X_train)
imputed_X_test=my_imputer.transform(X_test)
print('mean abs error from imputation:')
print(score_dataset(imputed_X_train,imputed_X_test,y_train,y_test))
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = [col for col in X_train.columns 
                                 if X_train[col].isnull().any()]
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer =Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))