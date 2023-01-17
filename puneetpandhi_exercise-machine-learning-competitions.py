# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



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



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
import matplotlib.pyplot as plt

import seaborn as sns



features1 = ['MSSubClass','LotArea','ExterQual' ,'TotalBsmtSF', 'YearBuilt','YearRemodAdd', 'GarageQual', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr','OverallQual','OverallCond','TotRmsAbvGrd','BsmtFinSF1','GrLivArea','Fireplaces','GarageArea','GarageCars','SalePrice']



X_new = home_data[features1]

X_first =home_data[features1]



#Changing varchar to Int for 2 critical columns

conditn_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

X_new['ExterQual'] = X_first['ExterQual'].map(conditn_map).astype('int')
#get correlations of each features in dataset

corrmat = X_new.corr()



top_corr_features = corrmat.index

plt.figure(figsize=(75,75))

#plot heat map

g=sns.heatmap(X_new[top_corr_features].corr(),annot=True,cmap="RdYlGn")
corrmat
corrmat['SalePrice'].sort_values(ascending=False).head(20)
features_final = ['OverallQual','GrLivArea','ExterQual', 'GarageArea','GarageCars','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','Fireplaces','BsmtFinSF1','LotArea', '2ndFlrSF',  'BedroomAbvGr']



X_new = home_data[features_final]

X_first = home_data[features_final]



X_new['ExterQual'] = X_first['ExterQual'].map(conditn_map).astype('int')





# To improve accuracy, create a new Random Forest model which you will train on all training data, notsure:

rf_model_on_full_data = RandomForestRegressor(random_state=1,max_leaf_nodes=1000)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X_new, y)
y.describe()
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)





# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features_final]

test_X_first = home_data[features_final]



#Handing Missing values in 2 cloumns & Coomuting INt to Char value

test_X = test_X.fillna(test_X.mean())

test_X['ExterQual'] = test_X_first['ExterQual'].map(conditn_map).astype('int')





# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)





# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.





output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
X_new.describe()
def get_random_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
#Wrote this function to test out which Leaf node size may work

for max_leaf_nodes in [5, 50, 500, 625, 750,1000, 2500, 5000]:

    my_mae = get_random_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# Check your answer

step_1.check()

# step_1.solution()