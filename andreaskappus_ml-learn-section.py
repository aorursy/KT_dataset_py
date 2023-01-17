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
test_X = test[data_predictors]

predicted_house_prices = my_model.predict(test_X)

print(predicted_house_prices)
print(data.isnull().sum()) # gets the sum of missing data for each column
new_data = data.copy() # thought i'd better copy data as starting new learning part
new_test = test.copy()
data_with_no_missing_values = new_data.dropna(axis=1) # drops values with missing values
columns_missing_values = [col for col in new_data.columns # iterates through columns and checks for null values
                          if new_data[col].isnull().any()]
reduced_data = new_data.drop(columns_missing_values, axis=1)
reduced_test_data = new_test.drop(columns_missing_values, axis=1)

from sklearn.preprocessing import Imputer
original_data = pd.read_csv('../input/train.csv')
my_imputer = Imputer() # initialize the Imputer
data = pd.DataFrame(my_imputer.fit_transform(original_data.select_dtypes(exclude=['object'])))
data.columns = original_data.select_dtypes(exclude = ['object']).columns
newer_data = data.copy() # make a copy

columns_missing_new = [col for col in newer_data.columns # iterates through columns and checks for null values
                          if newer_data[col].isnull().any()]

for col in columns_missing_new:
    newer_data[col + ' was_missing'] = newer_data[col].isnull()
    
my_imputer = Imputer()
data = pd.DataFrame(my_imputer.fit_transform(original_data.select_dtypes(exclude=['object'])))
newer_data.columns = original_data.select_dtypes(exclude = ['object']).columns
newer_data = my_imputer.fit_transform(newer_data)
iowa_data = pd.read_csv('../input/train.csv')

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

numeric_predictors = iowa_predictors.select_dtypes(exclude=['object']) # exclude object types so there's only numeric values
X_train, X_test, y_train, y_test = train_test_split(numeric_predictors, iowa_target,
                                                   train_size=0.7,test_size=0.2,random_state=0)

def score_data(X_train, X_test, y_train, y_test):
    my_model = RandomForestRegressor()
    my_model.fit(X_train, y_train)
    predictions = my_model.predict(X_test)
    return(mean_absolute_error(y_test, predictions))

columns_missing_values = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(columns_missing_values, axis=1) # reduce the train data by dropping data with missing values
reduced_X_test = X_test.drop(columns_missing_values, axis=1) # do the same for test data so train can more accurately compare

print("Mean Absolute Error after dropping columns missing values:")
print(score_data(reduced_X_train, reduced_X_test, y_train, y_test))
my_imputer = Imputer()

imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.fit_transform(X_test)

print("Mean Absolute Error after imputing:")
print(score_data(imputed_X_train, imputed_X_test, y_train, y_test))
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_data.dropna(axis=0, subset=['SalePrice'],inplace=True)

target = train_data.SalePrice

columns_missing_vals = [col for col in train_data.columns # iterate through the columns to find missing values
                       if train_data[col].isnull().any()]

new_train_predictors = train_data.drop(['Id', 'SalePrice'] + columns_missing_vals, axis=1) # drop Id and saleprice aswell as the columns missing values variable
new_test_predictors = test_data.drop(['Id'] + columns_missing_vals, axis=1)

low_cardinality_columns = [cname for cname in new_train_predictors.columns # cardinality means unique-ness, the loop iterates through the columns and checks for objects
                              if new_train_predictors[cname].nunique() < 10 and new_train_predictors[cname].dtype == 'object']
numeric_columns = [cname for cname in new_train_predictors.columns  
                      if new_train_predictors[cname].dtype in ['int64', 'float64']]

my_columns = low_cardinality_columns + numeric_columns # add the 2 new columns into the dataframe

main_train_pred = new_train_predictors[my_columns]
main_test_pred = new_test_predictors[my_columns]

main_train_pred.dtypes.sample(10) # object type indicates some text (like string)
one_hot_encoded_training_preds = pd.get_dummies(main_train_pred)
from sklearn.model_selection import cross_val_score
#mae = mean absolute error
def getmae(X,y):
    return -1 * cross_val_score(RandomForestRegressor(50), X, y, scoring = 'neg_mean_absolute_error').mean()
# multiply by -1 to return a positive mae score instead of negative value returned as sklearn convention

predictors_without_categoricals = main_train_pred.select_dtypes(exclude=['object'])

mae_without_categoricals = getmae(predictors_without_categoricals, target)

mae_one_hot_encoded = getmae(one_hot_encoded_training_preds, target)

print('mean absolute error after dropping categoricals: ' + str(int(mae_without_categoricals)))
print('mean absolute error with one hot encoding: ' + str(int(mae_one_hot_encoded)))

data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice # prediction target is SalesPrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object']) # predictor methods are all numerical data except saleprice

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25) # split training and test data

my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
from xgboost import XGBRegressor

my_model = XGBRegressor()

my_model.fit(train_X, train_y, verbose = False)
prediction = my_model.predict(test_X)

print("Mean absolute Error of XGBoost model is: " + str(mean_absolute_error(prediction, test_y)))
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.5)
my_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)] ,verbose=False)

prediction = my_model.predict(test_X)


print("Mean absolute Error of XGBoost model is: " + str(mean_absolute_error(prediction, test_y)))