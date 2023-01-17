# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.preprocessing import OneHotEncoder







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]

enc = OneHotEncoder(handle_unknown='ignore')

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

model2= GaussianNB()

clf = SVC(gamma='auto')



# Fit Model

iowa_model.fit(train_X, train_y)

model2.fit(train_X, train_y)

clf.fit(train_X, train_y) 

enc.fit(train_X)





results = {}

for i in range(1000, 3001, 100):

    model = RandomForestRegressor(n_estimators=i)

    model.fit(train_X, train_y)

    y_preds = model.predict(val_X)

    result = mean_absolute_error(y_true=val_y, y_pred=y_preds)

    

    print('mae for n_estimators={} is {}'.format(i, result))

    results[i] = result



print('best mae', min(results.values()))

best_est = sorted([(v, k) for k, v in results.items()])[0][1]

print('best estimator num', best_est)









# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

val_predictions_model2 = model2.predict(val_X)

val_mae_model2 = mean_absolute_error(val_predictions_model2, val_y)



val_predictions_clf = clf.predict(val_X)

val_mae_clf = mean_absolute_error(val_predictions_clf, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

print("val_mae_model2: {:,.0f}".format(val_mae_model2))

print("val_mae_clf: {:,.0f}".format(val_mae_clf))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

sns.barplot(x=home_data.index, y=home_data['LotArea'])

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(n_estimators=1000,random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_y)





# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

# yy=['Id','SalePrice']

# y = test_data[yy]



features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = test_data[features]



# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#test_X = val_X



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(X)





# MY_rf_val_mae = mean_absolute_error(y_pred=test_preds, y_true=X)

# print("Validation MAE for Random Forest Model: {:,.0f}".format(MY_rf_val_mae))



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)