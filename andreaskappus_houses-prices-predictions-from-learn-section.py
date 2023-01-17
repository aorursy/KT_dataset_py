import pandas as pd

main_file_path = '../input/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)
print(data.describe())

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
print('Some output from running this cell')
print(data.columns) # view all columns
price_data = data.SalePrice
print(price_data.head)
columns_of_interest = ['Neighborhood', 'Utilities']
two_cheeky_columns = data[columns_of_interest]
two_cheeky_columns.describe()
y = data.SalePrice
data_predictors = ['LotArea', 'YearBuilt','1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[data_predictors]
from sklearn.tree import DecisionTreeRegressor

my_model = DecisionTreeRegressor() # initialize model
my_model.fit(X,y) # fit the prediction target method(y) and predictors(X) to model
print("using psychic machine learning powers to predict following 5 houses")
print(X.head())
print("Predictions are...")
print(my_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error

predicted_home_prices = my_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
my_model = DecisionTreeRegressor()
my_model.fit(train_X, train_y)

# get predicted prices on validation data
value_predictions = my_model.predict(val_X)
print(mean_absolute_error(val_y, value_predictions))

#mae = Mean absolute error
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y) # loading my own values into function
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
data_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, data_preds))

test = pd.read_csv('../input/test.csv')

test_X = test[data_predictors] # may as well get vector from previous model to save typing
predicted_house_prices = my_model.predict(test_X)
print(predicted_house_prices) # not to be confused with "predicted_home_prices

my_submission = pd.DataFrame({'ID': test.Id, 'saleprice': predicted_house_prices})
my_submission.to_csv('submission.csv', index = False)