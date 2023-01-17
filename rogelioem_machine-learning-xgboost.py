import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error

# Load data
iowa_file_path = '../input/house-prices-advanced-regression-techniques/train.csv'
iowa_data = pd.read_csv(iowa_file_path) 

##Separating data in feature predictors, and target to predict
y = iowa_data.SalePrice
# For the sake of keeping the example simple, we'll use only numeric predictors. 
X = iowa_data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

### Generating training and test data
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.3)

## imputing NaN values
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

### building XGBoost model 
from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)
# make predictions
predictions = my_model.predict(test_X)

### Calculate the Mean absolute error of the model
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

### Tuning model
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=True)
# make predictions
predictions = my_model.predict(test_X)

### The fixed n_estimators = 329 was obtained by the number of updates previously obtained by the fitting of the XGBRegressor
new_model = XGBRegressor(n_estimators=329)
new_model.fit(train_X, train_y, verbose=False)
new_predictions = new_model.predict(test_X)

### Calculate the Mean absolute error of the models
print("Mean Absolute Error with learning rate and early stopping : " + str(mean_absolute_error(predictions, test_y)))
print("Mean Absolute Error without fixed n_estimators: " + str(mean_absolute_error(new_predictions, test_y)))