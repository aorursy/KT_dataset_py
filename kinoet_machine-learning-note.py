import pandas as pd

# path to dataset file
file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame
data = pd.read_csv(file_path) 
# print a summary of the data
data.describe()
# drop null data
data = data.dropna(axis=0)
# check column names
print(data.columns)
print('----')

# get prediction target, usually called 'y', but I prefer semantic name
target = data.Price
print(target.head())
print('----')
# choose features, usually called 'X', but I prefer semantic name
features = data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]
print(features.head())
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)
# Fit model
model.fit(features, target)

print("Prediction on training data:")
print(model.predict(features.head()))
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# in-sample validation
predictions = model.predict(features)
print("In-sample score:")
print(mean_absolute_error(target, predictions))

# split data into training and validation data, for both features and target
train_features, val_features, train_target, val_target = train_test_split(features, target, random_state = 0)
# Define model
model2 = DecisionTreeRegressor()
# Fit model
model2.fit(train_features, train_target)

# get predicted prices on validation data
val_predictions = model2.predict(val_features)
print("Real score:")
print(mean_absolute_error(val_target, val_predictions))
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_features, train_target)
forest_prediction = forest_model.predict(val_features)
print(mean_absolute_error(val_target, forest_prediction))