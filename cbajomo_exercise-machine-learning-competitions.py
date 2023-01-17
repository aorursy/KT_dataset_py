# Code you have previously used to load data

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from learntools.core import *





my_imputer = SimpleImputer()

# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

extra_features = ['YearRemodAdd','GarageCars','GarageArea','GarageYrBlt','Fireplaces','TotalBsmtSF'

                  , 'YearBuilt','OverallQual','FullBath','GrLivArea','TotRmsAbvGrd','MasVnrArea']

features = ['YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'] + extra_features

X = home_data[features]

#X = X.fillna(0.0)

X = pd.DataFrame(my_imputer.fit_transform(X),columns=features)

    

#,'GarageCars','GarageArea'



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

plt.figure(figsize=(30,27))

sns.clustermap(home_data.corr(),annot=True)
#home_data.info()

#home_data['FireplaceQu'].unique()

#home_data[['GarageCars','GarageArea','GarageYrBlt','Fireplaces','TotalBsmtSF', 'YearBuilt','OverallQual','FullBath','GrLivArea']].head()

#home_data['Neighborhood'].unique()

#home_data["HeatingQC"].unique()

#home_data['GarageQual'].unique()

#home_data['GarageCond'].unique()
X.head()
home_data["HeatingQC"].unique()

X['HeatingQC_Ex'] = home_data["HeatingQC"].apply(lambda x: 1 if x == "Ex" else 0)

X['HeatingQC_Gd'] = home_data["HeatingQC"].apply(lambda x: 1 if x == "Gd" else 0)

X['HeatingQC_TA'] = home_data["HeatingQC"].apply(lambda x: 1 if x == "TA" else 0)

X['HeatingQC_Fa'] = home_data["HeatingQC"].apply(lambda x: 1 if x == "Fa" else 0)



X['GarageQual_Ex'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Ex" else 0)

X['GarageQual_Gd'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Gd" else 0)

X['GarageQual_TA'] = home_data["GarageQual"].apply(lambda x: 1 if x == "TA" else 0)

#X['GarageQual_Fa'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Fa" else 0)

#X['GarageQual_Po'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Po" else 0)





X['GarageCond_Ex'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Ex" else 0)

X['GarageCond_Gd'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Gd" else 0)

X['GarageCond_TA'] = home_data["GarageQual"].apply(lambda x: 1 if x == "TA" else 0)

#X['GarageCond_Fa'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Fa" else 0)

#X['GarageCond_Po'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Po" else 0)



#Neighborhood

X['Neighborhood_CollgCr'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "CollgCr" else 0)

X['Neighborhood_Veenker'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "Veenker" else 0)

X['Neighborhood_Crawfor'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "Crawfor" else 0)

X['Neighborhood_NoRidge'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "NoRidge" else 0)

X['Neighborhood_Mitchel'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "Mitchel" else 0)

X['Neighborhood_Somerst'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "Somerst" else 0)

X['Neighborhood_NWAmes'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "NWAmes" else 0)

X['Neighborhood_BrkSide'] = home_data["Neighborhood"].apply(lambda x: 1 if x == "BrkSide" else 0)







X.head(10)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

# path to file you will use for predictions

test_data_path = '../input/home-data-for-ml-course/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



#test_data.head(10)[['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GarageCars','GarageArea']]



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[[ 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'] + extra_features]

test_X = pd.DataFrame(my_imputer.fit_transform(test_X),columns=features)

test_X['HeatingQC_Ex'] = test_data["HeatingQC"].apply(lambda x: 1 if x == "Ex" else 0)

test_X['HeatingQC_Gd'] = test_data["HeatingQC"].apply(lambda x: 1 if x == "Gd" else 0)

test_X['HeatingQC_TA'] = test_data["HeatingQC"].apply(lambda x: 1 if x == "TA" else 0)

test_X['HeatingQC_Fa'] = test_data["HeatingQC"].apply(lambda x: 1 if x == "Fa" else 0)



test_X['GarageQual_Ex'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Ex" else 0)

test_X['GarageQual_Gd'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Gd" else 0)

test_X['GarageQual_TA'] = home_data["GarageQual"].apply(lambda x: 1 if x == "TA" else 0)

#test_X['GarageQual_Fa'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Fa" else 0)

#test_X['GarageQual_Po'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Po" else 0)





test_X['GarageCond_Ex'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Ex" else 0)

test_X['GarageCond_Gd'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Gd" else 0)

test_X['GarageCond_TA'] = home_data["GarageQual"].apply(lambda x: 1 if x == "TA" else 0)

#test_X['GarageCond_Fa'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Fa" else 0)

#test_X['GarageCond_Po'] = home_data["GarageQual"].apply(lambda x: 1 if x == "Po" else 0)



test_X['Neighborhood_CollgCr'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "CollgCr" else 0)

test_X['Neighborhood_Veenker'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "Veenker" else 0)

test_X['Neighborhood_Crawfor'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "Crawfor" else 0)

test_X['Neighborhood_NoRidge'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "NoRidge" else 0)

test_X['Neighborhood_Mitchel'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "Mitchel" else 0)

test_X['Neighborhood_Somerst'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "Somerst" else 0)

test_X['Neighborhood_NWAmes'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "NWAmes" else 0)

test_X['Neighborhood_BrkSide'] = test_data["Neighborhood"].apply(lambda x: 1 if x == "BrkSide" else 0)



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)



output.head(10)