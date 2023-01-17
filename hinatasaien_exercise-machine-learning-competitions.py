# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

#from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Imputer

from sklearn.model_selection import cross_val_score

# Path of the file to read. We changed the directory structure to simplify submitting to a competition





iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)



y = home_data.SalePrice

X = home_data.drop(['SalePrice'],axis=1)



def prepro(input_data):

    #del_obj_data = input_data.select_dtypes(exclude=['object'])

    #print(del_obj_data.columns)

    #print(del_obj_data.columns)



    low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].nunique() < 10 and

                                candidate_train_predictors[cname].dtype == "object"]

    numeric_cols = [cname for cname in candidate_train_predictors.columns if 

                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]

    

    out_data = del_obj_data

    return out_data

X_pro = prepro(X)

my_pipeline =make_pipeline(Imputer(),RandomForestRegressor())

scores = cross_val_score(my_pipeline,X_pro,y,scoring='neg_mean_absolute_error')

print(scores)

my_pipeline.fit(X_pro, y)





test_data_path = '../input/test.csv'

test_data = pd.read_csv(test_data_path)

test_preds = my_pipeline.predict(prepro(test_data))





#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds})

#output.to_csv('submission.csv', index=False)

#print(output)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

print(output)