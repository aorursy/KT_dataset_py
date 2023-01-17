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

## find columns contain nulls 

col_nulls = home_data.columns[home_data.isna().any()].tolist()

print (col_nulls)
import operator

dic = {}

for val in col_nulls:

    dic[val] = [round((len(home_data[val].loc[home_data[val].isna()]) / len(home_data[val])) * 100,2), home_data[val].dtype]

print (sorted(dic.items(), key=operator.itemgetter(1)))    
## droping columns with 47% and above missing values

features_to_drop = ['FireplaceQu','Fence','Alley','MiscFeature','PoolQC']

for val in features_to_drop:

    col_nulls.remove(val)

    

home_data2 = home_data.drop(features_to_drop, axis=1)

print(home_data2.columns)
## replace missing values with mean value for LotFrontage since it makes almost 20% of the data as nulls 

home_data2["LotFrontage"].fillna(round(home_data2['LotFrontage'].mean(),2), inplace = True) 

home_data2["MasVnrArea"].fillna(round(home_data2['MasVnrArea'].mean(),2), inplace = True) 

home_data2["GarageYrBlt"].fillna(round(home_data2['GarageYrBlt'].mean(),2), inplace = True) 





## remove the missing rows in the rest of NaN feature since they make almost 5%

## home_data2 = home_data2.dropna() ## drop the na would not make it match the test dataset 

## fill na with the tag which has the mdiean number 

col_nulls2 = col_nulls.copy()

col_nulls2.remove("LotFrontage")

col_nulls2.remove("MasVnrArea")

col_nulls2.remove("GarageYrBlt")



print(col_nulls2)

for val in col_nulls2:

    tmp = home_data2[[val]].groupby([val]).size().reset_index()

    tmp.columns = [val,'count'] 

    midv = sorted(tmp['count'])[len(tmp['count']) // 2]

    filling_char = tmp[val].loc[ tmp['count'] == midv].tolist()[0]

    home_data2[val].fillna(filling_char, inplace = True)

## find columns contain nulls 

col_nulls3 = home_data2.columns[home_data2.isna().any()].tolist()

print (col_nulls3)



## get categorical variables 

cat_vars = home_data2.select_dtypes(include='object').columns.tolist()

home_data2[cat_vars[0:10]].head()
## label enconding all categorical varibles 

from sklearn import preprocessing

home_data2[cat_vars] = home_data2[cat_vars].apply(preprocessing.LabelEncoder().fit_transform)

home_data2.head()
liv = home_data2.columns.tolist()

# Create target object and call it y

y = home_data2.SalePrice

liv.remove('SalePrice')

# Create X

X = home_data2[liv]

#X = (X - X.mean()) / (X.max() - X.min()) # normalization 

X
from sklearn.linear_model import LassoCV

model_lasso = LassoCV(alphas = [0.5, 0.1, 0.05, 0.001, 0.0005]).fit(X, y)



# fit rf_model_on_full_data on all data from the training data

model_lasso.fit(X, y)
# To improve accuracy, create a new Random Forest model which you will train on all training data

# Define the model. Set random_state to 1

rf_model_on_full_data = RandomForestRegressor(min_samples_leaf = 3, min_samples_split = 3 ,n_estimators=400, random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)
import numpy as np

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data =  pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

features = liv #['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

test_X = test_data[features]



## prerpossing 

## replace missing values with mean value for LotFrontage since it makes almost 20% of the data as nulls 

test_X["LotFrontage"].fillna(round(test_X['LotFrontage'].mean(),2), inplace = True) 

test_X["MasVnrArea"].fillna(round(test_X['MasVnrArea'].mean(),2), inplace = True) 

test_X["GarageYrBlt"].fillna(round(test_X['GarageYrBlt'].mean(),2), inplace = True) 





## remove the missing rows in the rest of NaN feature since they make almost 5%

## home_data2 = home_data2.dropna() ## drop the na would not make it match the test dataset 

## fill na with the tag which has the mdiean number 

col_nulls2 = test_X.columns[test_X.isna().any()].tolist()

print('col_nulls2 =',col_nulls2)

#col_nulls2.remove("LotFrontage")

#col_nulls2.remove("MasVnrArea")

#col_nulls2.remove("GarageYrBlt")



for val in col_nulls2:

    tmp = test_X[[val]].groupby([val]).size().reset_index()

    tmp.columns = [val,'count'] 

    midv = sorted(tmp['count'])[len(tmp['count']) // 2]

    filling_char = tmp[val].loc[ tmp['count'] == midv].tolist()[0]

    test_X[val].fillna(filling_char, inplace = True)



### end prerpossing



## remove the missing rows in the rest of NaN feature since they make almost 5%

#test_X = test_X.dropna()





## label enconding all categorical varibles 

cat_vars = test_X.select_dtypes(include='object').columns.tolist()

print('cat_vars =',cat_vars)



from sklearn import preprocessing

test_X[cat_vars] = test_X[cat_vars].apply(preprocessing.LabelEncoder().fit_transform)

#test_X = (test_X - test_X.mean()) / (test_X.max() - test_X.min()) #normalization

#test_X['Utilities'].fillna(0, inplace = True) ## this issue has appeared after dataframe normilization 



#col_nulls4 = test_X.columns[test_X.isna().any()].tolist()





# make predictions which we will submit.

test_preds = rf_model_on_full_data.predict(test_X) 

pred_test = model_lasso.predict(test_X)

print('pred_test', pred_test)

print('test_preds', test_preds)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': pred_test})

output.to_csv('submission.csv', index=False)