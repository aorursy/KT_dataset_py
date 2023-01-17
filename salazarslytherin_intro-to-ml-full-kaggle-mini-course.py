import pandas as pd
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.describe())

print(melbourne_data.columns)
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

iowa_data = pd.read_csv(iowa_file_path)

print(iowa_data.describe())

print(iowa_data.columns)
# drops missing values

melbourne_data = melbourne_data.dropna(axis=0)



# target variable y

y = melbourne_data.Price



# features selected for judgement

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

print(X.describe())

print(X.head())
from sklearn.tree import DecisionTreeRegressor



melbourne_model = DecisionTreeRegressor(random_state=1) # random_state=1 to ensure same result at every run

melbourne_model.fit(X,y)



print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")

print(melbourne_model.predict(X.head()))
yy = iowa_data.SalePrice



# features selected for judgement

iowa_features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF","FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

XX = iowa_data[iowa_features]

print(XX.describe())

print(XX.head())



from sklearn.tree import DecisionTreeRegressor



iowa_model = DecisionTreeRegressor(random_state=1) # random_state=1 to ensure same result at every run

iowa_model.fit(XX,yy)



print("Making predictions for the following 5 houses:")

print(XX.head())

print("The predictions are")

print(iowa_model.predict(XX.head()))
from sklearn.metrics import mean_absolute_error



predicted_melbourne_prices = melbourne_model.predict(X)

print(mean_absolute_error(y, predicted_melbourne_prices))
from sklearn.model_selection import train_test_split





train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

melbourne_model = DecisionTreeRegressor()

melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
from sklearn.model_selection import train_test_split





train_X, val_X, train_y, val_y = train_test_split(XX, yy, random_state = 0)

iowa_model = DecisionTreeRegressor()

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return mae





melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

filtered_melbourne_data = melbourne_data.dropna(axis=0)

y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

X = filtered_melbourne_data[melbourne_features]

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
for max_leaf_nodes in [5, 50, 500, 5000]:

    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor



melbourne_forest_model = RandomForestRegressor(random_state=1)

melbourne_forest_model.fit(train_X, train_y)

melbourne_forest_preds = melbourne_forest_model.predict(val_X)

print(mean_absolute_error(val_y, melbourne_forest_preds))
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *



iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'

home_data = pd.read_csv(iowa_file_path)



y = home_data.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallQual', 'OverallCond']

X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(train_X, train_y)



val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



rf_model_on_full_data = RandomForestRegressor(random_state=1)

rf_model_on_full_data.fit(X,y)



test_data_path = '../input/house-prices-advanced-regression-techniques/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

test_preds = rf_model_on_full_data.predict(test_X)



output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)