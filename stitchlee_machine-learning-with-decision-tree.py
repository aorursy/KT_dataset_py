#Setting up environment

import pandas as pd



#Loading data

melbourne_file_path = '../input/Melbourne_housing_FULL.csv'

melbourne_data = pd.read_csv(melbourne_file_path) 

melbourne_data.columns
#Dropping missing values

melbourne_data = melbourne_data.dropna(axis=0)



#Setting up (x,y) coordinates

y = melbourne_data.Price



melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']



X = melbourne_data[melbourne_features]
#Reviewing data

X.head()
#Building Model

from sklearn.tree import DecisionTreeRegressor



# Define model

melbourne_model = DecisionTreeRegressor(random_state=1)



# Fit model

melbourne_model.fit(X, y)
#Making the predictions

print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")

print(melbourne_model.predict(X.head()))
#Validating the model

from sklearn.metrics import mean_absolute_error



predicted_home_prices = melbourne_model.predict(X)

mean_absolute_error(y, predicted_home_prices)