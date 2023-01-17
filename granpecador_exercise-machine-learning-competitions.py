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
# The list of columns is stored in a variable called features

features = [ 'MSSubClass',

'MSZoning',

'LotFrontage',

'LotArea',

'LotShape',

'LotConfig',

'Neighborhood',

'BldgType',

'HouseStyle',

'OverallQual',

'OverallCond',

'YearBuilt',

'YearRemodAdd',

'RoofStyle',

'Exterior1st',

'Exterior2nd',

'ExterQual',

'ExterCond',

'Foundation',

'BsmtQual',

'BsmtExposure',

'BsmtFinType1',

'BsmtFinSF1',

'BsmtUnfSF',

'TotalBsmtSF',

'HeatingQC',

'1stFlrSF',

'2ndFlrSF',

'GrLivArea',

'BsmtFullBath',

'FullBath',

'HalfBath',

'BedroomAbvGr',

'KitchenQual',

'TotRmsAbvGrd',

'Fireplaces',

'GarageType',

'GarageYrBlt',

'GarageFinish',

'GarageCars',

'GarageArea',

'WoodDeckSF',

'OpenPorchSF',

'YrSold']

home_data[features].info()
home_data['MSZoning'] = home_data['MSZoning'].map({'RH':1, 'RL':2, 'RM':3,'FV':4,'C (all)':5}).fillna(-1)

home_data['LotShape'] = home_data['LotShape'].map({'Reg':1, 'IR1':2, 'IR2':3,'IR3':4}).fillna(-1)

home_data['LotConfig'] = home_data['LotConfig'].map({'Inside':1, 'Corner':2, 'FR2':3,'CulDSac':4,'FR3':5}).fillna(-1)

home_data['Neighborhood'] = home_data['Neighborhood'].map({'NAmes':1, 'Gilbert':2, 'StoneBr':3, 'BrDale':4, 'NPkVill':5, 'NridgHt':6,

       'Blmngtn':7, 'NoRidge':8, 'Somerst':9, 'SawyerW':10, 'Sawyer':11, 'NWAmes':12,

       'OldTown':13, 'BrkSide':14, 'ClearCr':15, 'SWISU':16, 'Edwards':17, 'CollgCr':18,

       'Crawfor':19, 'Blueste':20, 'IDOTRR':21, 'Mitchel':22, 'Timber':23, 'MeadowV':24,

       'Veenker':25}).fillna(-1)

home_data['BldgType'] = home_data['BldgType'].map({'1Fam':1, 'TwnhsE':2, 'Twnhs':3,'Duplex':4,'2fmCon':5}).fillna(-1)

home_data['HouseStyle'] = home_data['HouseStyle'].map({'1Story':1, '2Story':2, 'SLvl':3,'1.5Fin':4,'SFoyer':5, '2.5Unf':6,'1.5Unf':7}).fillna(-1)

home_data['RoofStyle'] = home_data['RoofStyle'].map({'Gable':1, 'Hip':2, 'Gambrel':3,'Flat':4,'Mansard':5, 'Shed':6}).fillna(-1)

home_data['Exterior1st'] = home_data['Exterior1st'].map({'VinylSd':1, 'Wd Sdng':2, 'HdBoard':3, 'Plywood':4, 'MetalSd':5, 'CemntBd':6,

       'WdShing':7, 'BrkFace':8, 'AsbShng':9, 'BrkComm':10, 'Stucco':11, 'AsphShn':12,

       'CBlock':13}).fillna(-1)

home_data['Exterior2nd'] = home_data['Exterior2nd'].map({'VinylSd':1, 'Wd Sdng':2, 'HdBoard':3, 'Plywood':4, 'MetalSd':5, 'Brk Cmn':6,

       'CmentBd':7, 'ImStucc':8, 'Wd Shng':9, 'AsbShng':10, 'Stucco':11, 'CBlock': 12,                                

       'BrkFace':13, 'AsphShn':14,'Stone':15}).fillna(-1)

home_data['ExterQual'] = home_data['ExterQual'].map({'TA':1, 'Gd':2, 'Ex':3,'Fa':4}).fillna(-1)

home_data['ExterCond'] = home_data['ExterCond'].map({'TA':1, 'Gd':2, 'Fa':3,'Po':4, 'Ex':5}).fillna(-1)

home_data['Foundation'] = home_data['Foundation'].map({'CBlock':1, 'PConc':2, 'BrkTil':3,'Stone':4,'Slab':5, 'Wood':6}).fillna(-1)

home_data['BsmtQual'] = home_data['BsmtQual'].map({'TA':1, 'Gd':2, 'Ex':3,'Fa':4}).fillna(-1)

home_data['BsmtExposure'] = home_data['BsmtExposure'].map({'No':1, 'Gd':2, 'Mn':3,'Av':4}).fillna(-1)

home_data['BsmtFinType1'] = home_data['BsmtFinType1'].map({'Rec':1, 'ALQ':2, 'GLQ':3,'Unf':4,'BLQ':5, 'LwQ':6}).fillna(-1)

home_data['HeatingQC'] = home_data['HeatingQC'].map({'TA':1, 'Gd':2, 'Fa':3,'Po':4, 'Ex':5}).fillna(-1)

home_data['BsmtFullBath'] = home_data['BsmtFullBath'].fillna(-1)

home_data['KitchenQual'] = home_data['KitchenQual'].map({'TA':1, 'Gd':2, 'Ex':3,'Fa':4}).fillna(-1)

home_data['GarageType'] = home_data['GarageType'].map({'Attchd':1, 'Detchd':2, 'BuiltIn':3,'Basment':4,'2Types':5, 'CarPort':6}).fillna(-1)

home_data['GarageFinish'] = home_data['GarageFinish'].map({'Unf':1, 'Fin':2, 'RFn':3}).fillna(-1)

home_data['GarageCars'] = home_data['GarageCars'].fillna(-1)

home_data['LotFrontage'] = home_data['LotFrontage'].fillna(-1)

home_data['BsmtFinSF1'] = home_data['BsmtFinSF1'].fillna(-1)

home_data['BsmtUnfSF'] = home_data['BsmtUnfSF'].fillna(-1)

home_data['TotalBsmtSF'] = home_data['TotalBsmtSF'].fillna(-1)

home_data['GarageYrBlt'] = home_data['GarageYrBlt'].fillna(-1)

home_data['GarageArea'] = home_data['GarageArea'].fillna(-1)

home_data[features].info()
# dropna drops missing values (think of na as "not available")

#home_data = home_data.dropna(axis=0, how='all')



# Create target object and call it y

y = home_data.SalePrice

# Create X



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

rf_model = RandomForestRegressor(random_state=1, max_features =14)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1, max_features =14)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

import pandas as pd

test_data = pd.read_csv(test_data_path)

test_data.head()
test_data['MSZoning'] = test_data['MSZoning'].map({'RH':1, 'RL':2, 'RM':3,'FV':4,'C (all)':5}).fillna(-1)

test_data['LotShape'] = test_data['LotShape'].map({'Reg':1, 'IR1':2, 'IR2':3,'IR3':4}).fillna(-1)

test_data['LotConfig'] = test_data['LotConfig'].map({'Inside':1, 'Corner':2, 'FR2':3,'CulDSac':4,'FR3':5}).fillna(-1)

test_data['Neighborhood'] = test_data['Neighborhood'].map({'NAmes':1, 'Gilbert':2, 'StoneBr':3, 'BrDale':4, 'NPkVill':5, 'NridgHt':6,

       'Blmngtn':7, 'NoRidge':8, 'Somerst':9, 'SawyerW':10, 'Sawyer':11, 'NWAmes':12,

       'OldTown':13, 'BrkSide':14, 'ClearCr':15, 'SWISU':16, 'Edwards':17, 'CollgCr':18,

       'Crawfor':19, 'Blueste':20, 'IDOTRR':21, 'Mitchel':22, 'Timber':23, 'MeadowV':24,

       'Veenker':25}).fillna(-1)

test_data['BldgType'] = test_data['BldgType'].map({'1Fam':1, 'TwnhsE':2, 'Twnhs':3,'Duplex':4,'2fmCon':5}).fillna(-1)

test_data['HouseStyle'] = test_data['HouseStyle'].map({'1Story':1, '2Story':2, 'SLvl':3,'1.5Fin':4,'SFoyer':5, '2.5Unf':6,'1.5Unf':7}).fillna(-1)

test_data['RoofStyle'] = test_data['RoofStyle'].map({'Gable':1, 'Hip':2, 'Gambrel':3,'Flat':4,'Mansard':5, 'Shed':6}).fillna(-1)

test_data['Exterior1st'] = test_data['Exterior1st'].map({'VinylSd':1, 'Wd Sdng':2, 'HdBoard':3, 'Plywood':4, 'MetalSd':5, 'CemntBd':6,

       'WdShing':7, 'BrkFace':8, 'AsbShng':9, 'BrkComm':10, 'Stucco':11, 'AsphShn':12,

       'CBlock':13}).fillna(-1)

test_data['Exterior2nd'] = test_data['Exterior2nd'].map({'VinylSd':1, 'Wd Sdng':2, 'HdBoard':3, 'Plywood':4, 'MetalSd':5, 'Brk Cmn':6,

       'CmentBd':7, 'ImStucc':8, 'Wd Shng':9, 'AsbShng':10, 'Stucco':11, 'CBlock': 12,                                

       'BrkFace':13, 'AsphShn':14,'Stone':15}).fillna(-1)

test_data['ExterQual'] = test_data['ExterQual'].map({'TA':1, 'Gd':2, 'Ex':3,'Fa':4}).fillna(-1)

test_data['ExterCond'] = test_data['ExterCond'].map({'TA':1, 'Gd':2, 'Fa':3,'Po':4, 'Ex':5}).fillna(-1)

test_data['Foundation'] = test_data['Foundation'].map({'CBlock':1, 'PConc':2, 'BrkTil':3,'Stone':4,'Slab':5, 'Wood':6}).fillna(-1)

test_data['BsmtQual'] = test_data['BsmtQual'].map({'TA':1, 'Gd':2, 'Ex':3,'Fa':4}).fillna(-1)

test_data['BsmtExposure'] = test_data['BsmtExposure'].map({'No':1, 'Gd':2, 'Mn':3,'Av':4}).fillna(-1)

test_data['BsmtFinType1'] = test_data['BsmtFinType1'].map({'Rec':1, 'ALQ':2, 'GLQ':3,'Unf':4,'BLQ':5, 'LwQ':6}).fillna(-1)

test_data['HeatingQC'] = test_data['HeatingQC'].map({'TA':1, 'Gd':2, 'Fa':3,'Po':4, 'Ex':5}).fillna(-1)

test_data['BsmtFullBath'] = test_data['BsmtFullBath'].fillna(-1)

test_data['KitchenQual'] = test_data['KitchenQual'].map({'TA':1, 'Gd':2, 'Ex':3,'Fa':4}).fillna(-1)

test_data['GarageType'] = test_data['GarageType'].map({'Attchd':1, 'Detchd':2, 'BuiltIn':3,'Basment':4,'2Types':5, 'CarPort':6}).fillna(-1)

test_data['GarageFinish'] = test_data['GarageFinish'].map({'Unf':1, 'Fin':2, 'RFn':3}).fillna(-1)

test_data['GarageCars'] = test_data['GarageCars'].fillna(-1)

test_data['LotFrontage'] = test_data['LotFrontage'].fillna(-1)

test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(-1)

test_data['BsmtUnfSF'] = test_data['BsmtUnfSF'].fillna(-1)

test_data['TotalBsmtSF'] = test_data['TotalBsmtSF'].fillna(-1)

test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(-1)

test_data['GarageArea'] = test_data['GarageArea'].fillna(-1)

#test_data[features].info()
# create test_X which comes from test_data but includes only the columns you used for prediction.

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)