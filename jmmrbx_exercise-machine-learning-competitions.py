# Code you have previously used to load data

import pandas as pd

import numpy as np

#from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

#from sklearn.tree import DecisionTreeRegressor

from sklearn.feature_extraction import FeatureHasher

from sklearn.impute import SimpleImputer



from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor



import warnings

warnings.filterwarnings("ignore")



from learntools.core import *



def print_categorical_data_info(data_frame):

    for column in data_frame.select_dtypes(include=['object']).columns:

         print(column, ' :', len(data_frame[column].unique()))



def get_categorical_labels(data_frame, column):

    labels = data_frame[column].unique();

    if not('None' in labels):

        labels = np.append(labels, 'None')

    labels.sort()

    return labels



def fill_categorical_data(original_frame):



    data_frame = original_frame.copy()

    

    for column in data_frame.select_dtypes(include=['object']).columns:

         data_frame[column] = data_frame[column].fillna("None")

    return data_frame



def fill_non_categorial_data(original_frame):



    data_frame = original_frame.copy()

    

    missing_val_count_by_column = (data_frame[data_frame.select_dtypes(exclude=['object']).columns].isnull().sum())

    features = missing_val_count_by_column[missing_val_count_by_column > 0]



    if len(features) > 0:

        imputer = SimpleImputer()

        new_values= pd.DataFrame(imputer.fit_transform(data_frame[features.index]), columns=features.index)



        for column in data_frame[features.index]:

            data_frame[column] = new_values[column]

                  

    return data_frame



def fill_data(data_frame):

    return fill_non_categorial_data(fill_categorical_data(data_frame))



def clean_data(original_frame):



    data_frame = original_frame.copy()

    

    data_frame.drop(['ExterQual'], axis=1, inplace = True)

    data_frame.drop(['LotFrontage'], axis=1, inplace = True)

    data_frame.drop(['MSSubClass'], axis=1, inplace = True)

    data_frame.drop(['MSZoning'], axis=1, inplace = True)

    data_frame.drop(['BldgType'], axis=1, inplace = True)

    data_frame.drop(['BedroomAbvGr'], axis=1, inplace = True)

    data_frame.drop(['HouseStyle'], axis=1, inplace = True)

    data_frame.drop(['GarageCond'], axis=1, inplace = True)

    data_frame.drop(['MiscVal'], axis=1, inplace = True)

        

    return data_frame



def encode_lables(original_frame):



    data_frame = original_frame.copy()

    

    for column in data_frame.select_dtypes(include=['object']).columns:

        lables = get_categorical_labels(data_frame, column)

        feature_hasher = FeatureHasher(n_features=5, input_type='string')



        hashed_features = feature_hasher.fit_transform(data_frame[column], lables)

        new_data = pd.DataFrame(hashed_features.toarray())



        new_data.rename(columns = lambda x: column + '_' + str(x) , inplace=True)

        data_frame.drop([column], axis=1, inplace=True)



        data_frame = pd.concat([data_frame, new_data], axis=1)



    return data_frame



def get_model(model_name):

    if model_name == 'RandomForestRegressor':

        model = RandomForestRegressor(random_state=5)

    elif model_name == 'XGBRegressor':

        model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

    elif model_name == 'LinearRegression':

           model = LinearRegression()

    elif model_name == 'Lasso':

        model = Lasso(alpha=0.0005, random_state=5, max_iter=10000)

    elif model_name == 'Ridge':

        model = Ridge(alpha=0.002, random_state=5)

    elif model_name == 'ElasticNet':

        model = ElasticNet(alpha=0.02, random_state=5, l1_ratio=0.7)

    elif model_name == 'KNeighborsRegressor':

        model = KNeighborsRegressor()

    elif model_name == 'GradientBoostingRegressor':

        model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=5)

    elif model_name == 'SVR':

        model = SVR(kernel='linear')

     

    return model



def test_model(model_name,train_X, val_X, train_y, val_y):

    model = get_model(model_name)

    model.fit(train_X, train_y)

    val_predictions = model.predict(val_X)

    val_mae = mean_absolute_error(val_predictions, val_y)



    return val_mae



def test_model_all(train_X, val_X, train_y, val_y):

    models = ['RandomForestRegressor', 'XGBRegressor', 'LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'KNeighborsRegressor', 'GradientBoostingRegressor', 'SVR']

#    models = ['RandomForestRegressor', 'XGBRegressor',  'GradientBoostingRegressor']

    i = 0

    model_results = dict()

    print()

    print('test_model_all')        

    for model_name in models:

        model_results[model_name] = test_model(model_name,train_X, val_X, train_y, val_y)



    return model_results



def test_drop_one(original_home_data):



    original_features = original_home_data.columns.drop(['Id', 'SalePrice'])

    #original_features = ['ExterQual', 'LotFrontage', 'MSSubClass', 'MSZoning', 'BldgType', 'BedroomAbvGr', 'GarageCond']

    

    for ignore_feature in original_features:

        print()

        print('---Test ignore feature: ', ignore_feature) 

        new_data = original_home_data.copy()

        home_data = fill_data(new_data)



        home_data.drop(ignore_feature, axis=1, inplace = True)

        home_data = encode_lables(home_data)

        #features = original_features.drop(ignore_feature)



        y = home_data.SalePrice

        X = home_data[home_data.columns.drop(['Id', 'SalePrice'])]

         

        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



        model_results = test_model_all(train_X, val_X, train_y, val_y)

        print_model_results(model_results)

        print('End Test')

        

def print_model_results(model_results):

    for model_name in model_results:

        print("Validation MAE for " + model_name +": {:,.0f}".format(model_results[model_name]))







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

#iowa_file_path = './data/train.csv'



# path to file you will use for predictions

test_data_path = '../input/test.csv'

#test_data_path = './data/test.csv'



# read test data file using pandas

#v0

#home_data = pd.read_csv(iowa_file_path)

#test_data = pd.read_csv(test_data_path)



#v2

original_home_data = pd.read_csv(iowa_file_path)

original_test_data = pd.read_csv(test_data_path)



original_home_data = clean_data(original_home_data)

original_test_data = clean_data(original_test_data)



home_data = encode_lables(fill_data(original_home_data))

test_data = encode_lables(fill_data(original_test_data))



# Create target object and call it y

y = home_data.SalePrice



#Missing values 

#v0

#missing_val_count_by_column = (home_data.isnull().sum())



#v1

missing_val_count_by_column_home = (home_data[home_data.select_dtypes(exclude=['object']).columns].isnull().sum())

missing_val_count_by_column_test = (test_data[test_data.select_dtypes(exclude=['object']).columns].isnull().sum())



#print("Columns with missing values")

#print(missing_val_count_by_column_home[missing_val_count_by_column_home == 0])

#print(missing_val_count_by_column_test[missing_val_count_by_column_test == 0])



# Create X



#v0

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



#v2

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'OverallQual']

features = (missing_val_count_by_column_home[missing_val_count_by_column_home == 0].index.intersection(missing_val_count_by_column_test[missing_val_count_by_column_test == 0].index)).drop('Id')



# LotFrontage 

# Imputation

# LotFrontage



#from sklearn.impute import SimpleImputer

#my_imputer = SimpleImputer()



#data_with_imputed_values = my_imputer.fit_transform(home_data[features])



# Create X



#X = data_with_imputed_values

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

# v.1 Now probably not the best value anymore 

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



print()

print('base')

       

model_results = test_model_all(train_X, val_X, train_y, val_y)

print_model_results(model_results)



#test_drop_one(original_home_data)
# To improve accuracy, create a new Random Forest model which you will train on all training data

# rf_model_on_full_data =  RandomForestRegressor(random_state=1)

rf_model_on_full_data = get_model('XGBRegressor')

# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)





# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit.

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('sample_submission.csv', index=False)
