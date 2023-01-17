# Code you have previously used to load data

import pandas as pd

import numpy as np



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline



from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

import xgboost as xgb











# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)





    

# path to file you will use for predictions

test_data_path = '../input/test.csv'

# read test data file using pandas

test_data = pd.read_csv(test_data_path)
#Option 1: Select certain columns 



features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

y = home_data.SalePrice

X = np.array(home_data[features])



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)


#Option 2: Select only numeric columns without nulls 



def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)





# Load data

y = home_data.SalePrice







#Categorical + Numerical

#home_data_option = pd.get_dummies(home_data)

#test_data_option = pd.get_dummies(test_data)





#Only Numerical 

home_data_option = home_data.select_dtypes(exclude=['object'])

test_data_option = test_data.select_dtypes(exclude=['object'])





home_data_option_predictors = home_data_option.drop(['SalePrice','Id'], axis=1)

test_data_option_predictors = test_data_option.drop(['Id'], axis=1)



#home_data_option_predictors_encoded, test_data_option_predictors_encoded = home_data_option_predictors.align(test_data_option,join='inner', axis=1)



cols_with_missing = [col for col in home_data_option_predictors.columns if home_data_option_predictors[col].isnull().any()]

X = home_data_option_predictors.drop(cols_with_missing, axis=1)

test_X = test_data_option_predictors.drop(cols_with_missing, axis=1)





print("Train Data Columns: " + str(len(X.columns)))

print("Train Test Columns: " + str(len(test_X.columns)))





train_X, val_X, train_y, val_y  = train_test_split( np.array(X), y,test_size = 0.2,  random_state=42)
d_train = xgb.DMatrix(data = train_X, 

                       label = train_y)  

d_valid =  xgb.DMatrix(data = val_X,

                       label = val_y)



##Head

print(train_X)



##Numeric columns list

print(len(train_X))

print('\n\n\n');

my_pipeline = Pipeline([('imputer', SimpleImputer()), ('xgbrg', xgb.XGBRegressor())])
param_grid = {

    "xgbrg__n_estimators": [100,200, 500,1000,1500],

    "xgbrg__learning_rate": [0.01,0.05,0.1, 0.5],

    "xgbrg__max_depth": [10, 15, 20, 25],

    'xgbrg__colsample_bytree': np.linspace(0.5, 0.9, 5)

}



fit_params = {"xgbrg__eval_set": [(val_X, val_y)], 

              "xgbrg__early_stopping_rounds": 5, 

              "xgbrg__verbose": False} 



gridSearchCV = GridSearchCV(my_pipeline, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 1, n_jobs = -1)

#gridSearchCV = GridSearchCV(my_pipeline, cv=5, param_grid=param_grid)

gridSearchCV.fit(train_X, train_y,**fit_params)  


## option 1





param_grid = {

    "xgbrg__n_estimators": [100,200, 500,1000,1500],

    "xgbrg__learning_rate": [0.01,0.05,0.1, 0.5],

    "xgbrg__max_depth": [10, 15, 20, 25],

    'xgbrg__colsample_bytree': np.linspace(0.5, 0.9, 5)

}

fit_params = {"xgbrg__eval_set": [(val_X, val_y)], 

              "xgbrg__early_stopping_rounds": 5, 

              "xgbrg__verbose": False} 



randomizedSearchCV = RandomizedSearchCV(my_pipeline, param_distributions = param_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

randomizedSearchCV.fit(train_X, train_y,**fit_params)
optionCV = gridSearchCV
print(optionCV.best_params_)
#Best Params Repository



# gridSearchCV 

#{'xgbrg__colsample_bytree': 0.5, 'xgbrg__learning_rate': 0.01, 'xgbrg__max_depth': 10, 'xgbrg__n_estimators': 1500}







#randomizedSearchCV 

#{'xgbrg__n_estimators': 1000, 'xgbrg__max_depth': 10, 'xgbrg__learning_rate': 0.01, 'xgbrg__colsample_bytree': 0.5}

#

#{'xgbrg__n_estimators': 1500, 'xgbrg__max_depth': 10, 'xgbrg__learning_rate': 0.05, 'xgbrg__colsample_bytree': 0.5}

#

#{'xgbrg__colsample_bytree': 0.5, 'xgbrg__learning_rate': 0.05, 'xgbrg__max_depth': 10, 'xgbrg__n_estimators': 500}
def evaluate(model, test_features, test_labels):

    predictions = model.predict(test_features)

    errors = abs(predictions - test_labels)

    mape = 100 * np.mean(errors / test_labels)

    accuracy = 100 - mape

    print('Model Performance')

    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))

    print('Accuracy = {:0.2f}%.'.format(accuracy))

    print(' ')

    return accuracy





xgb_params = {

        'learning_rate': 0.05,

        'n_estimators': 500,

        'max_depth':10,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'silent': 1,

        'verbose': False,

        'seed': 27}

#base_model = xgb.XGBRegressor(param_grid=xgb_params, num_boost_round = 10000 )

#base_model.fit(train_X, train_y)

#base_accuracy = evaluate(base_model,   val_X, val_y  )





    



base_model = xgb.train(xgb_params, d_train, num_boost_round = 10000, evals=[(d_valid, 'eval')], verbose_eval=100, 

                     early_stopping_rounds=100)

base_accuracy = evaluate(base_model,   val_X, val_y  )





  

#best_cv = optionCV.best_estimator_

#random_accuracy = evaluate(best_cv,  val_X, val_y  )





#print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# To improve accuracy, create a new Random Forest model which you will train on all training data

model_on_full_data =  optionCV.best_estimator_





# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features



#Option 1: Basic Column Selection

#featuresTest = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#test_X = test_data[featuresTest]



#Option 2: Only Numeric Columns

test_X = test_data_option_predictors

test_X = test_X.fillna(test_X.mean())



d_test = xgb.DMatrix(test_X)



test_preds = model_on_full_data.predict(d_test)







print(test_preds)

# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.





output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

print('CSV GENERATED !!!')