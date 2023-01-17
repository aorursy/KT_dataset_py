# 1st we need to find out coloumns with null
#2nd make imputer class and compare others missing value
#3rd Add col with missing values to your predictor

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

#load data
lowa_data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
lowa_target = lowa_data.SalePrice
lowa_predictors = lowa_data.drop(['SalePrice'], axis=1) # drops col 
#print(lowa_predictors) # print 80 coloumns instead of 81 
#lowa_predictors = lowa_data.drop(25, axis=1) #drops row
lowa_predictors_n = lowa_predictors.select_dtypes(exclude=['object']) 
#print(lowa_predictors_n)# print 37 coloumns instead of 81 because it dropped all non numeric coloumns

#Bring column column tells how many not null in it
#lowa_predictors_n.info()
#score_dataset Create Function to Measure Quality of An Approach
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(lowa_predictors_n, lowa_target, train_size=0.7, test_size=0.3,  random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# 1st we need to find out coloumns with null
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
#2) Use the Imputer class so you can impute missing values
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
#3) Add columns with missing values to your predictors. 
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
