# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor



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
rf_model = RandomForestRegressor(random_state=1)#state 7 works better than 1 (only 16 points though)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

help(train_X[:1].isnull().any)
candidate_train_X.describe()
"""
The MAE differs according to run
"""
def get_mae(test_model,train_X,train_y,val_X,val_y):
    """
    function for checking the mean absolute error of a 
    given model
    """
    test_model.fit(train_X,train_y)
    pred_vals = test_model.predict(val_X)
    mae = mean_absolute_error(val_y,pred_vals)
    return mae

#using more features for construction
home_data.dropna(axis=0,subset=['SalePrice'],inplace=True)
iowa_target = home_data.SalePrice

iowa_predictors = home_data.drop(['SalePrice'],axis=1)

#dropping non-numeric data
#iowa_numerics = iowa_predictors.select_dtypes(exclude=['object'])

candidate_train_X,candidate_val_X,train_y,val_y = train_test_split(iowa_predictors,iowa_target,random_state=1)

#candidate_train_X.drop(['Id'],axis=1)
#candidate_val_X.drop(['Id'],axis=1)

low_cardinality_cols = [cname for cname in candidate_train_X.columns if
                       candidate_train_X[cname].nunique()<10 and
                       candidate_train_X[cname].dtype=='object']
numeric_cols = [cname for cname in candidate_train_X.columns if
               candidate_train_X[cname].dtype in ['int64','float64']] 

my_cols = low_cardinality_cols+numeric_cols
ctrain_X = candidate_train_X[my_cols]
cval_X = candidate_val_X[my_cols]

#one-hot-encoding
one_hot_encoded_training_predictors = pd.get_dummies(ctrain_X)
one_hot_encoded_test_predictors = pd.get_dummies(cval_X)
train_X,val_X = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                     join='inner',
                                                                     axis=1)

cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]

reduced_train_X = train_X.drop(cols_with_missing,axis=1)
reduced_val_X = val_X.drop(cols_with_missing,axis = 1)

test_model = RandomForestRegressor()
reduced_train_X = train_X.drop(cols_with_missing,axis=1)
reduced_val_X = val_X.drop(cols_with_missing,axis=1)

print("Mean absolute Error from dropping columns with Missing Values")
print(get_mae(test_model,reduced_train_X,train_y,reduced_val_X,val_y))

"""imputations"""

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_train_X =  my_imputer.fit_transform(train_X)
imputed_val_X = my_imputer.transform(val_X)
print("Mean Absolute Error from Imputation:")
print(get_mae(test_model,imputed_train_X,train_y,imputed_val_X,val_y))


copy_train_X = train_X.copy()
copy_val_X = val_X.copy()

for col in cols_with_missing:
    copy_train_X[col+'_was_missing'] = copy_train_X[col].isnull()
    copy_val_X[col+'_was_missing'] = copy_val_X[col].isnull()
#imputation with id
copy_train_X = my_imputer.fit_transform(copy_train_X)
copy_val_X = my_imputer.fit_transform(copy_val_X)
print("Mean Absolute Error from Imputation while Tracking it")
print(get_mae(test_model,copy_train_X,train_y,copy_val_X,val_y))
# To improve accuracy, create a new Random Forest model which you will train on all training data
#rf_model_on_full_data = RandomForestRegressor()

# fit rf_model_on_full_data on all data from the 
#rf_model_on_full_data.fit(X,y)

test_data.Alley
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
home_data = pd.read_csv(iowa_file_path)

#using more features for construction
home_data.dropna(axis=0,subset=['SalePrice'],inplace=True) #drop houses with no saleprice
iowa_target = home_data.SalePrice #get saleprices remaining
iowa_predictors = home_data.drop(['SalePrice'],axis=1) #drop houses from predictors 

#dropping non-numeric data
#test_numerics = test_data.select_dtypes(exclude=['object'])
#iowa_numerics = iowa_predictors.select_dtypes(exclude=['object'])

low_cardinality_cols = [cname for cname in iowa_predictors.columns 
                        if iowa_predictors[cname].nunique()<10 
                        and iowa_predictors[cname].dtype=='object']

numeric_cols = [cname for cname in iowa_predictors.columns
               if iowa_predictors[cname].dtype in ['int64','float64']]

my_cols = low_cardinality_cols+numeric_cols
ctrain_X = iowa_predictors[my_cols] 
ctest_X = test_data[my_cols]

#one_hot_encoding
one_hot_train = pd.get_dummies(ctrain_X)
one_hot_test = pd.get_dummies(ctest_X)

#alignment
train_X,test_X = one_hot_train.align(one_hot_test,join='inner',axis=1)

#imputation
my_imputer = SimpleImputer()

copy_train = train_X.copy()
copy_test = test_X.copy()

#find columns with missing values
cols_with_missing = [col for col in copy_train.columns if copy_train[col].isnull().any()]

for col in cols_with_missing:
    copy_train[col+"_was_missing"] = copy_train[col].isnull()
    copy_test[col+"_was_missing"] = copy_test[col].isnull()

copy_train = my_imputer.fit_transform(copy_train)
copy_test = my_imputer.fit_transform(copy_test)

rf_model_on_full_data = RandomForestRegressor()
rf_model_on_full_data.fit(copy_train,iowa_target)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
#test_X = test_data[features]

# make predictions which we will submit.
test_preds = rf_model_on_full_data.predict(copy_test)

#print("Mean Absolute Error from Imputation while Tracking it")
#print(get_mae(rf_model_on_full_data,copy_train,iowa_target,copy_test,test_data_target))

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id':test_data.Id,'SalePrice':test_preds})
output.to_csv('submission.csv',index=False)