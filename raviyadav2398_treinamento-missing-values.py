import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
house_pricing = pd.read_csv('../input/home-data-for-ml-course/train.csv')
house_pricing.head()
s = (house_pricing.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
house_pricing.drop(object_cols, inplace = True, axis = 1)
house_pricing.head()
house_pricing.info()
y = house_pricing.SalePrice
y.head()
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = house_pricing.copy()
X.drop(['SalePrice'], axis=1, inplace=True)
X.head()
X.info()
from sklearn.model_selection import train_test_split 
X_train, X_test,y_train, y_test, = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
missing_values_columns = (house_pricing.isnull().sum())
print(missing_values_columns[missing_values_columns>0])
missing_values_columns = (X_train.isnull().sum())
print(missing_values_columns[missing_values_columns>0])
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def scores_mae(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators = 100, random_state=1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return mean_absolute_error(y_test, pred)
#Drop columns with missing values

cols_missing = [col for col in X_train.columns if X_train[col].isnull().any()]               
reduc_X_train = X_train.drop(cols_missing, axis=1)
reduc_X_test = X_test.drop(cols_missing, axis=1)

reduc_X_train.head()
print("MAE (Drop columns with missing values):")
print(scores_mae(reduc_X_train, reduc_X_test, y_train, y_test))
#Imputation

from sklearn.impute import SimpleImputer

impute = SimpleImputer()
impute_X_train = pd.DataFrame(impute.fit_transform(X_train)) 
impute_X_test = pd.DataFrame(impute.transform(X_test))

impute_X_train.columns = X_train.columns
impute_X_test.columns = X_test.columns
print("MAE (Imputation):")
print(scores_mae(impute_X_train, impute_X_test, y_train, y_test))
model = RandomForestRegressor(n_estimators = 100, random_state=1)
model.fit(reduc_X_train, y_train)
pred = model.predict(reduc_X_test)
# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': reduc_X_test.index,
                       'SalePrice': pred})
output.to_csv('submision.csv', index=False)
output.head()