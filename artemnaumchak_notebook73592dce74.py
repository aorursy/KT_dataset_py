import numpy as np

import pandas as pd



# Read the data

train_full = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col="Id")

test_full = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col="Id")



# Remove rows with missing target, separate target from predictors

train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)



print("Setup complete")
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer



def prepare_data(train, valid, test=None):



    # All categorical columns

    object_cols = [cname for cname in train.columns if train[cname].dtype == "object"]

    # Columns that will be one-hot encoded

    low_cardinality_cols = [col for col in object_cols if train[col].nunique() < 20]

    # Columns that will be dropped from the dataset

    high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))



    #print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)

    #print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols)

    

    obj_train = train[low_cardinality_cols].fillna('None')

    obj_valid = valid[low_cardinality_cols].fillna('None')



    oh_Encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    OH_cols_train = pd.DataFrame(oh_Encoder.fit_transform(obj_train))

    OH_cols_valid = pd.DataFrame(oh_Encoder.transform(obj_valid))



    OH_cols_train.index = train.index

    OH_cols_valid.index = valid.index



    # Fill in the lines below: imputation

    numeric_imputer = SimpleImputer(strategy="mean")



    no_obj_train = train.drop(object_cols, axis=1).drop(['SalePrice'], axis=1)

    no_obj_valid = valid.drop(object_cols, axis=1).drop(['SalePrice'], axis=1)



    num_train = pd.DataFrame(numeric_imputer.fit_transform(no_obj_train))

    num_valid = pd.DataFrame(numeric_imputer.transform(no_obj_valid))



    num_train.index = train.index

    num_valid.index = valid.index



    # Fill in the lines below: imputation removed column names; put them back

    num_train.columns = no_obj_train.columns

    num_valid.columns = no_obj_valid.columns



    X_train = pd.concat([num_train, OH_cols_train], axis=1)

    y_train = train['SalePrice']



    X_valid = pd.concat([num_valid, OH_cols_valid], axis=1)

    y_valid = valid['SalePrice']

        

    if test is not None: 

        obj_test = test[low_cardinality_cols].fillna('None')

        oh_test = pd.DataFrame(oh_Encoder.transform(obj_test))

        oh_test.index = test.index

        

        no_obj_test = test.drop(object_cols, axis=1)

        

        num_test = pd.DataFrame(numeric_imputer.transform(no_obj_test))

        num_test.index = test.index

        num_test.columns = no_obj_test.columns

        

        X_test = pd.concat([num_test, oh_test], axis=1)

        

        return X_train, y_train, X_valid, y_valid, X_test

    else:

        return X_train, y_train, X_valid, y_valid



def get_data_splits(dataframe, valid_fraction=0.1):



    dataframe = dataframe.sort_index()

    valid_rows = int(len(dataframe) * valid_fraction)

    train = dataframe[:-valid_rows]

    valid = dataframe[-valid_rows:]

    

    return train, valid
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



def select_features_l1(X, y):

    """Return selected features using logistic regression with an L1 penalty."""

    logistic = LogisticRegression(C=0.5, penalty="l1", solver='liblinear', random_state=7).fit(X, y)

    selector = SelectFromModel(logistic, prefit=True)



    X_new = selector.transform(X) 



    # Get back the kept features as a DataFrame with dropped columns as all 0s

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                     index=X.index, 

                                     columns=X.columns)



    selected_columns = selected_features.columns[selected_features.var() != 0]

    return selected_columns



#print("Feature selection")



#n_samples = 300

#X, y = p_train[feature_cols][:n_samples], p_train["SalePrice"][:n_samples]

#selected = select_features_l1(X, y)



#dropped_columns = feature_cols.drop(selected)



#selected_features_train = p_train.drop(dropped_columns, axis=1)

#selected_features_valid = p_valid.drop(dropped_columns, axis=1)



#print('Selected features:', list(selected_features_train.columns))
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV



print("Finding best parameters")



train, valid = get_data_splits(train_full)

X_train, y_train, X_valid, y_valid = prepare_data(train, valid)



xgbr = XGBRegressor()



reg_cv = GridSearchCV(xgbr, {"min_child_weight":[0.8, 1.0, 1.2], "learning_rate": [0.1],

                             'max_depth': [2, 3, 8], 'n_estimators': [500]}, verbose=1)



reg_cv.fit(X_train, y_train)



print("Best parameters: ",reg_cv.best_params_)
from sklearn import metrics

from sklearn.model_selection import cross_val_score



def train_model(X_train, y_train, X_valid, y_valid, **kwargs):

    xgbr = XGBRegressor(**kwargs)

    

    #print("Training model")

    xgbr.fit(X_train, y_train)



    score = xgbr.score(X_train, y_train)  

    print("Training score: ", score)

    

    scores = cross_val_score(xgbr, X_train, y_train, cv=10)

    print("Mean cross-validation score: %.2f" % scores.mean())



    pred_valid = xgbr.predict(X_valid)

    #print(valid_pred)

    valid_score = metrics.mean_squared_error(y_valid, pred_valid, squared=False)

    #print(f"Validation mean score: {valid_score}")



    return xgbr, valid_score
print("Getting final results")



train, valid = get_data_splits(train_full)

X_train, y_train, X_valid, y_valid, X_test = prepare_data(train, valid, test_full)



regressor, final_valid_score = train_model(X_train, y_train, X_valid, y_valid, **reg_cv.best_params_)



print("Final validation score: ", final_valid_score)

# make predictions which we will submit. 

test_preds = regressor.predict(X_test)



# The lines below shows how to save predictions in format used for competition scoring

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)



print("Results saved")