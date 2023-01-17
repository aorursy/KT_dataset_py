# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
#for data imputation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
# path to file you will use for predictions
test_data_path = '../input/test.csv'

home_data = pd.read_csv(iowa_file_path)
# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# Create target object and call it y
y = home_data.SalePrice

X = home_data

#-Add more features with categorical parameters, but columns with null parameters should be excluded
cols_with_object_missing = [col for col in X.columns 
                                 if X[col].isnull().any() and X[col].dtype == "object"] 
#exclude Id, SalePrice and columns with null parameters
X = X.drop(['Id','SalePrice'] + cols_with_object_missing,axis = 1)

features = X.columns

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

#encode Categorical columns in data set X and test_X
X_buff = pd.get_dummies(X)
test_X_buff = pd.get_dummies(test_X)
X_buff, test_X_buff = X_buff.align(test_X_buff,join='left',axis=1)

# #---------pipeline---------#
XGBReg_on_full_data_pipeline = make_pipeline(SimpleImputer(),XGBRegressor())
#XGBReg_on_full_data_pipeline.fit(X, y)
# #---------pipeline---------#


#X_buff = X.drop(drop_feature,axis = 1)
scores = cross_val_score(XGBReg_on_full_data_pipeline, X_buff, y, scoring='neg_mean_absolute_error')
#print(drop_feature)
#print("full features:",scores)
mean_min = (-1 * scores.mean())
feature_of_no_use = ''

for drop_feature in features:   
    X_buff = X.drop(drop_feature,axis = 1)
    test_X_buff = test_X.drop(drop_feature,axis = 1)
    
    #encode Categorical columns in data set X and test_X
    X_buff = pd.get_dummies(X_buff)
    test_X_buff = pd.get_dummies(test_X_buff)
    X_buff, test_X_buff = X_buff.align(test_X_buff,join='left',axis=1)

    # #---------pipeline---------#
    XGBReg_on_full_data_pipeline = make_pipeline(SimpleImputer(),XGBRegressor())
    #XGBReg_on_full_data_pipeline.fit(X, y)
    # #---------pipeline---------#


    #X_buff = X.drop(drop_feature,axis = 1)
    scores = cross_val_score(XGBReg_on_full_data_pipeline, X_buff, y, scoring='neg_mean_absolute_error')
    #print("drop_feature:  ",drop_feature)
    #print(scores)
    if (-1 * scores.mean())<mean_min:
        mean_min = (-1 * scores.mean())
        feature_of_no_use = drop_feature
print("feature of no use:  ",feature_of_no_use)
print("the minimum mean:  ",mean_min)
# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
#for data imputation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
# path to file you will use for predictions
test_data_path = '../input/test.csv'

home_data = pd.read_csv(iowa_file_path)
# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# Create target object and call it y
y = home_data.SalePrice
X = home_data

#-Add more features with categorical parameters, but columns with null parameters should be excluded
cols_with_object_missing = [col for col in X.columns 
                                 if X[col].isnull().any() and X[col].dtype == "object"] 
#exclude Id, SalePrice and columns with null parameters
X = X.drop(['Id','SalePrice'] + cols_with_object_missing,axis = 1)

X = X.drop(['MoSold'],axis = 1)

features = X.columns

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

#encode Categorical columns in data set X and test_X
X = pd.get_dummies(X)
test_X = pd.get_dummies(test_X)
X, test_X = X.align(test_X,join='left',axis=1)

# #---------pipeline---------#
XGBReg_on_full_data_pipeline = make_pipeline(SimpleImputer(),XGBRegressor())
XGBReg_on_full_data_pipeline.fit(X, y)
# #---------pipeline---------#

# #---------pipeline---------#
#test_preds = cross_val_predict(XGBReg_on_full_data_pipeline,test_X,cv=3)
test_preds = XGBReg_on_full_data_pipeline.predict(test_X)
# #---------pipeline---------#

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
print("successful!")
# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
#for data imputation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
# path to file you will use for predictions
test_data_path = '../input/test.csv'

home_data = pd.read_csv(iowa_file_path)
# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# Create target object and call it y
y = home_data.SalePrice

# Create X
#the following two lines are the original code
# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# X = home_data[features]

#change the original code to cope with missing values
X = home_data#.select_dtypes(exclude = ['object'])
# X = X.drop(['SalePrice'],axis = 1)
# X = X.drop(['Id'],axis = 1)
#-Add more features with categorical parameters, but columns with null parameters should be excluded
cols_with_object_missing = [col for col in X.columns 
                                 if X[col].isnull().any() and X[col].dtype == "object"] 
#exclude Id, SalePrice and columns with null parameters
X = X.drop(['Id','SalePrice'] + cols_with_object_missing,axis = 1)

features = X.columns


# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

#encode Categorical columns in data set X and test_X
X = pd.get_dummies(X)
test_X = pd.get_dummies(test_X)
X, test_X = X.align(test_X,join='left',axis=1)

#-test 1-#drop columns with missing values
# cols_with_missing = [col for col in X.columns 
#                                  if X[col].isnull().any()]
# reduced_train_X = X.drop(cols_with_missing, axis=1) 
# X = reduced_train_X
#we only need to drop lines with missing parameters in the train data set
#then the columns remained are saved in the features, which would be recalled in the test data set
#-test 1-result: the result shows it does no good. The other column in the test data set contains null data

# #---------could be replaced by pipeline---------#
# #-test 2-#imputation
# my_imputer = SimpleImputer()
# imputed_train_X = my_imputer.fit_transform(X)
# X = imputed_train_X
# #impute the test data set
# imputed_test_X = my_imputer.transform(test_X)
# test_X =imputed_test_X
# X = pd.DataFrame(X)
# test_X = pd.DataFrame(test_X)
# #---------could be replaced by pipeline---------#

#------pipeline-------#
DecisionTreeReg_pipeline = make_pipeline(SimpleImputer(),DecisionTreeRegressor(max_leaf_nodes=100, random_state=1))
RandomForestReg_pipeline = make_pipeline(SimpleImputer(),RandomForestRegressor(random_state=1))
XGBReg_pipeline = make_pipeline(SimpleImputer(),XGBRegressor())
#------pipeline-------#

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#------pipeline-------#
DecisionTreeReg_pipeline.fit(train_X, train_y)
predictions = DecisionTreeReg_pipeline.predict(val_X)
val_mae = mean_absolute_error(predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

RandomForestReg_pipeline.fit(train_X, train_y)
predictions = RandomForestReg_pipeline.predict(val_X)
rf_val_mae = mean_absolute_error(predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

XGBReg_pipeline.fit(train_X, train_y)
predictions = XGBReg_pipeline.predict(val_X)
XGBoost_val_mae = mean_absolute_error(predictions, val_y)
print("Validation MAE for XGBoost Model: {:,.0f}".format(XGBoost_val_mae))
#------pipeline-------#

# #---------could be replaced by pipeline---------#
# # Specify Model
# iowa_model = DecisionTreeRegressor(random_state=1)
# iowa_model.fit(train_X, train_y)
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(val_predictions, val_y)
# print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# # Using best value for max_leaf_nodes
# iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
# iowa_model.fit(train_X, train_y)
# val_predictions = iowa_model.predict(val_X)
# val_mae = mean_absolute_error(val_predictions, val_y)
# print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# # Define the model. Set random_state to 1
# rf_model = RandomForestRegressor(random_state=1)
# rf_model.fit(train_X, train_y)
# rf_val_predictions = rf_model.predict(val_X)
# rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
# print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# #use the XGBoost model
# XGBoost_model = XGBRegressor()
# XGBoost_model.fit(train_X, train_y, verbose=False)
# XGBoost_var_predictions = XGBoost_model.predict(val_X)
# XGBoost_val_mae = mean_absolute_error(XGBoost_var_predictions, val_y)
# print("Validation MAE for XGBoost Model: {:,.0f}".format(XGBoost_val_mae))
# #---------could be replaced by pipeline---------#



#help(home_data.drop)
#help(DecisionTreeRegressor.fit)

# #---------could be replaced by pipeline---------#
# # To improve accuracy, create a new Random Forest model which you will train on all training data
# rf_model_on_full_data = RandomForestRegressor(random_state = 1)
# # fit rf_model_on_full_data on all data from the 
# rf_model_on_full_data.fit(X,y)

# # To improve accuracy, create a new XGBoost model which you will train on all training data
# XGBoost_model_on_full_data = XGBRegressor(n_estimators=500)
# # fit XGBoosy_model_on_full_data on all data from the 
# XGBoost_model_on_full_data.fit(X,y)
# #---------could be replaced by pipeline---------#

# #---------pipeline---------#
XGBReg_on_full_data_pipeline = make_pipeline(SimpleImputer(),XGBRegressor())
XGBReg_on_full_data_pipeline.fit(X, y)
# #---------pipeline---------#
print(X)
print(features)
print(len(features))

scores = cross_val_score(XGBReg_on_full_data_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)
    

# print(X)
# X_buff = X.drop(['MSSubClass'],axis = 1)
# print(X_buff)
#help(cross_val_score)
#help(cross_val_predict)

# # path to file you will use for predictions
# test_data_path = '../input/test.csv'

# # read test data file using pandas
# test_data = pd.read_csv(test_data_path)

# # create test_X which comes from test_data but includes only the columns you used for prediction.
# # The list of columns is stored in a variable called features
# test_X = test_data[features]

# #impute the test data set
# imputed_test_X = my_imputer.fit_transform(test_X)
# test_X =imputed_test_X

# make predictions which we will submit.
#the original code uses the random forest, now the model is changed into XGBoost to validation its advantages
#test_preds = rf_model_on_full_data.predict(test_X)

# #---------could be replaced by pipeline---------#
#test_preds = XGBoost_model_on_full_data.predict(test_X)
# #---------could be replaced by pipeline---------#

# #---------pipeline---------#
#test_preds = cross_val_predict(XGBReg_on_full_data_pipeline,test_X,cv=3)
test_preds = XGBReg_on_full_data_pipeline.predict(test_X)
# #---------pipeline---------#

# # The lines below shows you how to save your data in the format needed to score it in the competition
# output = pd.DataFrame({'Id': test_data.Id,
#                        'SalePrice': test_preds})

# output.to_csv('submission.csv', index=False)
# print("successful!")



