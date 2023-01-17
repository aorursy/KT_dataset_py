import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

from learntools.core import *



iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)



one_hot_encoded_home_data = pd.get_dummies(home_data)

one_hot_encoded_home_data  = one_hot_encoded_home_data.dropna(axis=0, subset=['MasVnrArea','LotFrontage','GarageYrBlt'])

y=one_hot_encoded_home_data.SalePrice 







dataset=one_hot_encoded_home_data
dataset=dataset.drop(['SalePrice'],axis=1)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

one_hot_encoded_test_data = pd.get_dummies(test_data)



final_train, final_test = dataset.align(one_hot_encoded_test_data,join='left',axis=1)
X=final_train.drop(['Id'],axis=1)
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
#missing_val_count_by_column = (imputed_test_data.isnull().sum())

#print(missing_val_count_by_column[missing_val_count_by_column > 0])
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(oob_score = True, n_jobs = -1,random_state =50,max_features = "auto", min_samples_leaf = 50)                                        



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_y)

# path to file you will use for predictions

#test_data_path = '../input/test.csv'



# read test data file using pandas

#test_data = pd.read_csv(test_data_path)



test_data
#test_data=test_data.drop(['Id'], axis=1)
#test_data=pd.get_dummies(test_data)
test_data= final_test.drop(['Id'], axis=1)
final_test
missing_val_count_by_column = (k.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
test_datafill=test_data
k=test_datafill
k
test_datafill.update(test_datafill[['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea','Utilities_NoSeWa','Condition2_RRAe','Condition2_RRAn','Condition2_RRNn','HouseStyle_2.5Fin','RoofMatl_ClyTile','RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Exterior1st_ImStucc','Exterior1st_Stone','Exterior2nd_Other','Heating_Floor','Heating_OthW','Electrical_Mix','GarageQual_Ex','PoolQC_Fa','MiscFeature_TenC']].fillna(0))
k.fillna(k.mean(), inplace=True)
k
# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X =k



# make predictions which we will submit. 

test_preds = rf_model.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

output = pd.DataFrame({'Id': final_test.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)