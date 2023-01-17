#This Kernel is done after going through all the ML courses on Kaggle Learn

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import make_pipeline



from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



from sklearn.preprocessing import Imputer

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
# List of all the functions that we are going to use



def get_rmse(y_predicted,y_real):

    return np.mean(np.sqrt((np.log(y_predicted)-np.log(y_real))**2))
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



y = train_data.SalePrice
train_data.head()

train_data.columns
#One-hot encoding (using categorical data)



cols_with_missing = [col for col in train_data.columns 

                                 if train_data[col].isnull().any()]                                  

candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)

candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)



# "cardinality" means the number of unique values in a column.

# We use it as our only way to select categorical columns here. This is convenient, though

# a little arbitrary.

low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].nunique() < 10 and

                                candidate_train_predictors[cname].dtype == "object"]

numeric_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

my_cols = low_cardinality_cols + numeric_cols

train_predictors = candidate_train_predictors[my_cols]

test_predictors = candidate_test_predictors[my_cols]



one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)
X = np.array(final_train)
train_X, val_X, train_y, val_y = train_test_split(X, y)
my_pipeline = Pipeline([('imputer', Imputer()), ('xgbrg', XGBRegressor())])



param_grid = {

    "xgbrg__n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 80],

    "xgbrg__learning_rate": [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3],

}



fit_params = {"xgbrg__eval_set": [(val_X, val_y)], 

              "xgbrg__early_stopping_rounds": 10, 

              "xgbrg__verbose": False} ;



#5-fold cross validation by passing the argument cv=5

searchCV = GridSearchCV(my_pipeline, cv=5, param_grid=param_grid, fit_params=fit_params);

searchCV.fit(train_X, train_y)  ;
searchCV.best_params_ 
best_learn_rate = searchCV.best_params_['xgbrg__learning_rate']

best_nb_est = searchCV.best_params_ ['xgbrg__n_estimators']
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=best_nb_est, learning_rate = best_learn_rate))



my_pipeline.fit(X,y)

train_y_predicted = my_pipeline.predict(train_X)

val_y_predicted = my_pipeline.predict(val_X)

print('Score on training set:',get_rmse(train_y_predicted,train_y))

print('Score on validation set:',get_rmse(val_y_predicted,val_y))
predictions = my_pipeline.predict(final_test)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': predictions})



output.to_csv('submission.csv', index=False)