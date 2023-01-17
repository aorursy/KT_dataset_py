import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence,partial_dependence
from matplotlib import pyplot as plt
# from sklearn.tree import DecisionTreeRegressor

iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice
num = ['int64', 'float64']
nope = ['Id', 'SalePrice']
features = [col for col in home_data.columns if home_data[col].dtype in num and col not in nope]
X = home_data[features]
X.describe()
# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Imputing values for missing fields
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
val_X = my_imputer.transform(val_X)

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

# Using XGBoost to create model
xg_model = XGBRegressor()
xg_model.fit(train_X, train_y)
xg_predict = xg_model.predict(val_X)
xg_mae = mean_absolute_error(xg_predict, val_y)

print("Validation MAE for XGBoost Model: {:,.0f}".format(xg_mae))
print("Validation MAE for RF Model: {:,.0f}".format(rf_val_mae))
my_model = GradientBoostingRegressor()
my_model.fit(train_X, train_y)

fig ,axs = plot_partial_dependence(my_model,
                                      features=[i for i in range(21)],
                                      X=train_X,
                                      feature_names=features,
                                      grid_resolution=20)

plt.subplots_adjust(top=3.5,right=3.5)
# To improve accuracy, create a new Random Forest model which you will train on all training data
xg_model_on_full_data = XGBRegressor()
X = my_imputer.fit_transform(X)

# fit rf_model_on_full_data on all data from the 
xg_model_on_full_data.fit(X,y)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
# print(features)
test_X = test_data[features]
test_X = my_imputer.fit_transform(test_X)

# make predictions which we will submit. 
test_preds = xg_model_on_full_data.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)