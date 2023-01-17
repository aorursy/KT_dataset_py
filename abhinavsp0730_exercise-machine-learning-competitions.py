# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
import matplotlib.pyplot as plt

test = []

for i in range(100,110):

    iowa_model = DecisionTreeRegressor(max_leaf_nodes=i,random_state=1)

    iowa_model = iowa_model.fit(train_X, train_y)

    score = iowa_model.score(val_X,  val_y)

    val_predictions = iowa_model.predict(val_X)

    val_mae = mean_absolute_error(val_predictions, val_y)

    print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

    test.append(val_mae)

plt.plot(range(100,110),test,color = "red",label = "max_leaf_nodes")

plt.legend()

plt.show()
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_y)



# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']





# The list of columns is stored in a variable called features

test_X = test_data[features]





# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# path to file you will use for predictions

from sklearn.preprocessing import Imputer

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

test_X = test_data[features]

my_imputer=Imputer(missing_values='NaN',strategy='median',axis=1)

imputed_test_X = my_imputer.transform(test_X)
#import numpy as np

#import pandas as pd

#imputed_test_X.describe()

#imputed_test_X
# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features





# make predictions which we will submit. 

test_preds = rf_model.predict(imputed_test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('./Predictions.csv', index=False)
#test_preds = rf_model_on_full_data.predict(imputed_test_X)

#test_preds