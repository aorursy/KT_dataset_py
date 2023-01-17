import pandas as pd

main_file_path = '../input/housetrain.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print(data.describe())
print(data.columns)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

y = data.SalePrice
predict_column = ['1stFlrSF','YearBuilt','KitchenAbvGr', 'BedroomAbvGr','LotArea']
X = data[predict_column]
print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
prediction = forest_model.predict(val_X)
print("Mean Absolute Error:  %d" %(mean_absolute_error(val_y, prediction)))
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.preprocessing import Imputer

y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors. 
numeric_X = X.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(numeric_X, y,random_state = 0)

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

#lets see columns with missing values
print(imputed_X_train_plus.isnull().sum())

# two numeric columns have a lot of missing values
imputed_X_train_plus['LotFrontage_was_missing'] = imputed_X_train_plus['LotFrontage'].isnull()
imputed_X_train_plus['GarageYrBlt_was_missing'] = imputed_X_train_plus['GarageYrBlt'].isnull()

imputed_X_test_plus['LotFrontage_was_missing'] = imputed_X_test_plus['LotFrontage'].isnull()
imputed_X_test_plus['GarageYrBlt_was_missing'] = imputed_X_test_plus['GarageYrBlt'].isnull()

# lets impute the missing values in the data set
my_imputer = Imputer()

imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

forest_model.fit(imputed_X_train_plus, y_train)
prediction = forest_model.predict(imputed_X_test_plus)
print("Mean Absolute Error:  %d" %(mean_absolute_error(y_test, prediction)))
