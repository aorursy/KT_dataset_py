# Code you have previously used to load data

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor





# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error





reg_model = RandomForestRegressor(n_estimators = 100, random_state = 0)



reg_model.fit(train_X, train_y)
val_X_pred = reg_model.predict(val_X)
val_X_pred
mean_absolute_error(val_y, val_X_pred)
# Path of the file to read

iowa_file_path1 = '../input/home-data-for-ml-course/test.csv'

test_data = pd.read_csv(iowa_file_path1)

test_data = test_data[features]
test_y_pred = reg_model.predict(test_data)
test_y_pred