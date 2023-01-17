import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# data manipulation
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# feature selection & hyperparameter tuning
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# ML models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
def preprocess_all_at_once(drop_na_cols = 0, categorical = 0, intercept=False, normalize=False):
    '''
    This function handles the preprocessing for this dataset.
    categorical input:
        - 0 : Remove categories from input data.
        - 1 : Leaves categories as is.
        - 2 : Use one hot encoding on data.
    drop_na_cols:
        - 0: Drops columns with NaNs.
        - 1: Leaves NaNs as is.
        - 2: Imputes columns with NaNs (if categories left as is, the impute will skip the columns).
    intercept:
        - True: Add a y-intercept in the training data.
        - False: Will not add the y-intercept.
        
    The preprocessing includes:
     - loading in the data (training and testing data)
     - discards useless features (ones that have a small number of values)
     - handles categorical data (either drops or encodes)
     - handles NaNs and Infs (either drops or imputes)
     - normalizes data
    '''
    # load in data
    x_data = pd.read_csv("../input/trainFeatures.csv")
    y_data = pd.read_csv("../input/trainLabels.csv")
    x_test = pd.read_csv("../input/testFeatures.csv")
    
    y_data.set_index('ids', inplace=True)
    x_data.set_index("ids", inplace=True)
    x_test.set_index('ids', inplace=True)
    
    # get rid of the title for the dataframes' indexes
    del x_data.index.name
    del y_data.index.name
    
    # empty or close-to-empty features
    useless_features = ["exclude", "ExcludeFromLists", "Rpt_Comp_Emp", "Incomplete", "DonorAdvisoryText", 
                "Direct_Support", "Indirect_Support", "Int_Expense","Depreciation", "Assets_45", 
                "Assets_46", "Assets_47c", "Assets_48c", "Assets_49", "Assets_54", "Liability_60",
                "Reader2_Date", "StatusID", "DonorAdvisoryDate", "ResultsID", "DonorAdvisoryCategoryID"]

    # add the categorical data to the features-to-drop list
    if categorical==0:
        useless_features.append('erkey')
        
    x_data.drop(useless_features, axis=1, inplace=True)
    x_test.drop(useless_features, axis=1, inplace=True)

    # get rid of rows with no y value in our training data
    useless_rows = y_data.index[y_data['OverallScore'].isnull()].tolist()
    x_data.drop(useless_rows, inplace=True)
    y_data.drop(useless_rows, inplace=True)
        
    # one hot encoding increases the number of features to some 2700 features, making it unwieldy.
    # Therefore if we include it, we will make sure to reduce the feature count using feature selection.
    if categorical==2:
        x_data = pd.get_dummies(x_data)
        x_test = pd.get_dummies(x_test)
        x_data, x_test = x_data.align(x_test, axis=1, fill_value=0)
        
    # deal with missing values
    if drop_na_cols == 0:
        missing1 = [col for col in x_data.columns if x_data[col].isnull().any()]
        missing2 = [col for col in x_test.columns if x_test[col].isnull().any()]
        cols_with_missing = np.union1d(missing1, missing2)

        x_data.drop(cols_with_missing, axis=1, inplace=True)
        x_test.drop(cols_with_missing, axis=1, inplace=True)
        
    elif drop_na_cols == 2:
        my_imputer = SimpleImputer()
        if categorical == 1:
            categorical_x_data = x_data['erkey']
            categorical_x_test = x_test['erkey']
            x_data.drop(["erkey"], axis=1, inplace=True)
            x_test.drop(["erkey"], axis=1, inplace=True)

        x_data = pd.DataFrame(my_imputer.fit_transform(x_data), columns = x_data.columns, index=x_data.index)
        x_test = pd.DataFrame(my_imputer.fit_transform(x_test), columns = x_test.columns, index=x_test.index)
    
    # normalize data
    if normalize:
        x_data = (x_data - x_data.mean()) / x_data.std()
        x_test = (x_test - x_test.mean()) / x_test.std()

        # If a column had 0 variance, the normalization brought it to NaN.
        # To handle this, we will impute/drop-NaNs again.
        # This is a chicken and egg problem, as one hot encoding, imputing, and normalization, each depends on the other.
        if drop_na_cols != 1:
            missing1 = [col for col in x_data.columns if x_data[col].isnull().any()]
            missing2 = [col for col in x_test.columns if x_test[col].isnull().any()]
            cols_with_missing = np.union1d(missing1, missing2)

            x_data.drop(cols_with_missing, axis=1, inplace=True)
            x_test.drop(cols_with_missing, axis=1, inplace=True)

    if categorical==1:  # if we leave the categorical data as is, now (after normalization and imputing) we add it back
        x_data = x_data.join(categorical_x_data)
        x_test = x_test.join(categorical_x_test)
            
    # adding a column of ones for the y-intercept
    if intercept:
        x_data.insert(0, 'y-intercept', 1)
        x_test.insert(0, 'y-intercept', 1)
        
    return x_data, y_data, x_test
def forwardSubsetSelection(model, max_feature_count, x_data, y_data, known_features = []):
    '''
    This function performs forward subset selection on x_data when
    trained with the given 'model'. 
    
    This model returns the names of the max_feature_count most 
    beneficial features in the data.
    '''
    features_to_use = known_features
    features_not_added = x_data.columns
    features_not_added.drop(known_features)
    
    for i in range(max_feature_count - len(known_features)):
        best_score = 0
        best_feature = 0
        for feature in features_not_added:
            tmp_features_vec = features_to_use + [feature]
            cur_score = cross_val_score(model, x_data[tmp_features_vec], y_data, cv=5, n_jobs=-1).mean()
            if cur_score > best_score:
                best_score = cur_score
                best_feature = feature
        print('Feature number', i+ len(known_features), 'added:', best_feature)
        features_to_use.append(best_feature)
        features_not_added.drop(best_feature)
    return features_to_use
lin_reg_model = Ridge()

known_features = ['AuditedFinancial_status', 'StaffList_Status', 'erkey_er-31529', 'erkey_er-30808', 'erkey_er-31998', 'erkey_er-32426', 'erkey_er-31525', 'Form990_status', 'erkey_er-31238', 'BoardList_Status', 'RatingTableID', 'CNVersion', 'CEO_Salary', 'Fundraising_Expenses', 'Program_Expenses', 'Privacy_Status', 'erkey_er-30284', 'erkey_er-30112', 'erkey_er-30839', 'erkey_er-31310', 'erkey_er-30453', 'erkey_er-31688', 'erkey_er-32312', 'erkey_er-31748', 'erkey_er-30539', 'erkey_er-32449', 'erkey_er-31455', 'erkey_er-31594', 'erkey_er-32517', 'erkey_er-32028', 'erkey_er-32207', 'erkey_er-30905', 'erkey_er-31615', 'erkey_er-31082', 'erkey_er-31676', 'erkey_er-31192', 'erkey_er-31545', 'erkey_er-31767', 'erkey_er-31769', 'erkey_er-31204', 'erkey_er-31892', 'erkey_er-30007', 'erkey_er-32603', 'erkey_er-31509', 'erkey_er-32341', 'erkey_er-30311', 'RatingInterval', 'erkey_er-32420', 'erkey_er-31327', 'erkey_er-30467', 'erkey_er-30774', 'erkey_er-32183', 'erkey_er-31899', 'erkey_er-32488', 'erkey_er-32432', 'erkey_er-30176', 'erkey_er-30958', 'erkey_er-31978', 'erkey_er-32455', 'erkey_er-30734', 'erkey_er-32305', 'erkey_er-30992', 'erkey_er-30890', 'erkey_er-31691', 'erkey_er-31461', 'erkey_er-30327', 'erkey_er-31271', 'erkey_er-32081', 'erkey_er-31617', 'erkey_er-32334', 'erkey_er-32274', 'erkey_er-31404', 'erkey_er-30371', 'erkey_er-31054', 'erkey_er-31019', 'erkey_er-30929', 'erkey_er-31363', 'erkey_er-30415', 'erkey_er-31986', 'erkey_er-30005', 'Total_Liabilities', 'Total_Net_Assets', 'Other_Revenue', 'erkey_er-30094', 'erkey_er-31255', 'erkey_er-31128', 'erkey_er-32297', 'erkey_er-30471', 'erkey_er-32499', 'erkey_er-30534', 'erkey_er-30824', 'erkey_er-32402', 'erkey_er-30609', 'erkey_er-31225', 'erkey_er-31859', 'Total_Expenses', 'Total_Revenue', 'erkey_er-30104', 'erkey_er-30373', 'erkey_er-30173', 'erkey_er-32521', 'erkey_er-32080', 'erkey_er-30928', 'erkey_er-30827', 'erkey_er-31827', 'erkey_er-31572', 'erkey_er-30765', 'erkey_er-32071', 'erkey_er-31966', 'erkey_er-32380', 'erkey_er-30510', 'erkey_er-32032', 'erkey_er-31205', 'Govt_Grants', 'erkey_er-31694', 'erkey_er-32154', 'erkey_er-30426', 'erkey_er-30782', 'erkey_er-32112', 'erkey_er-31444', 'erkey_er-30847', 'erkey_er-32691', 'erkey_er-32093', 'erkey_er-32523', 'erkey_er-31576', 'erkey_er-31924', 'erkey_er-30722', 'erkey_er-32240', 'erkey_er-31552', 'erkey_er-31646', 'MemDues', 'erkey_er-30058', 'erkey_er-31464', 'erkey_er-31138', 'erkey_er-31700', 'erkey_er-31416', 'erkey_er-31418', 'erkey_er-31073', 'erkey_er-31115', 'erkey_er-31290', 'erkey_er-30686', 'erkey_er-32430', 'erkey_er-30185', 'erkey_er-31867', 'erkey_er-30273', 'erkey_er-31156', 'erkey_er-31289', 'erkey_er-31709', 'erkey_er-32555', 'erkey_er-31643', 'erkey_er-31743', 'erkey_er-32306', 'erkey_er-30146', 'erkey_er-30082', 'erkey_er-31173', 'erkey_er-31329', 'erkey_er-32279', 'erkey_er-32457', 'erkey_er-30204', 'erkey_er-31303']

# features_to_use = forwardSubsetSelection(lin_reg_model, 160, x_data, y_data['OverallScore'], known_features)
# new_x_data = x_data[features_to_use]

# print(cross_val_score(lin_reg_model, new_x_data, y_data['OverallScore'], cv=5, n_jobs=-1).mean())
# print(features_to_use)
def dimensionalityReduce(model, feature_count, x_data, y_data):
    '''
    This function performs the TruncatedSVD on a given
    the given model trained on x_data and y_data. 
    
    This model returns the names of the max_feature_count most 
    beneficial features in the data.
    '''
    svd = TruncatedSVD(n_components=feature_count)  
    return svd.fit_transform(x_data, y_data)
def searchDimensions(x_data, y_data):
    '''
    Given input data to train on, this function runs through
    all the possible dimensions SVD can reduce x_data's features
    into and scores them, returning the feature count that achieves
    the best score through k-fold cross validation.
    '''
    feat_count = np.arange(1202,x_data.shape[1])
    
    best_count = 0
    best_score = 0
    lin_reg_model = Ridge(alpha=575)
    for count in feat_count:
        new_x = dimensionalityReduce(lin_reg_model, count, x_data, y_data)
        tmp_model = Ridge(alpha=575)
        tmp_score = cross_val_score(tmp_model, new_x, y_data, cv=5, n_jobs=-1).mean()
        if tmp_score > best_score:
            best_score = tmp_score
            best_count = count
            print('New best dimension: %d, score: %f'%(best_count, best_score))
    return best_count
def cross_val(model, vals, x_data, y_data):
    '''
    This function finds the optimal hyperparameter value from
    the 'vals' list, for the specified 'model' trained on x_data and y_data.
    '''
    best_val = 0
    best_score = 0
    for val in vals:
        our_model = model(val)
        tmp_score = cross_val_score(our_model, x_data, y_data, cv=5, n_jobs=-1).mean()
        if tmp > best_score:
            best_score = tmp_score
            best_val = val
            print('New best mse: %f for hyperparameter=%f'%(best_score, best_val))

    print('Best value found:', best_val)
    return best_val
# finding optimal n_estimators hyperparameter
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=False, normalize=False)
x_data, y_data = shuffle(x_data, y_data)

# n_estimators_options = np.arange(170, 250, 5)
# optimal_random_forest_val = cross_val(RandomForestRegressor, n_estimators_options, x_data, y_data['OverallScore'])
# submitting a random forest to output csv file
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=False, normalize=False)
x_data, y_data = shuffle(x_data, y_data)

optimal_random_forest_val = 172
random_forest_model = RandomForestRegressor(n_estimators=optimal_random_forest_val)
random_forest_model.fit(x_data, y_data['OverallScore'])
random_forest_predictions = random_forest_model.predict(testing_values)
random_forest_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': random_forest_predictions})
random_forest_output.to_csv('random_forest_result.csv', index=False)
xgboost_model = XGBRegressor()

# A parameter grid for XGBoost
params = {
            'min_child_weight': [12, 16],
            'learning_rate': [0.01, 0.03],
            'gamma': [1, 3],
            'n_estimators': [600, 800],
            'colsample_bytree': [0.5, 0.7],
            'max_depth': [15, 25]
         }

# clf = GridSearchCV(xgboost_model, params, n_jobs=-1, cv=2, verbose=2, refit=True, return_train_score=True)
# clf.fit(x_data, y_data['OverallScore'])
# clf.best_params_
# submitting the xgboost model to output csv file
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=False, normalize=False)
x_data, y_data = shuffle(x_data, y_data)

xgboost_model = XGBRegressor(colsample_bytree=0.8,  gamma=0.5, learning_rate=0.02, max_depth=18, min_child_weight=13, n_estimators=1000)
xgboost_model.fit(x_data, y_data['OverallScore'])
xgboost_predictions = xgboost_model.predict(testing_values)
xgboost_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': xgboost_predictions})
xgboost_output.to_csv('xgboost_result.csv', index=False)
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=True, normalize=True)
x_data, y_data = shuffle(x_data, y_data)

# finding optimal alpha hyperparameter
#alpha_options = np.arange(1, 1000, 5)
#optimal_ridge_val = cross_val(Ridge, alpha_options, x_data, y_data['OverallScore'])
optimal_ridge_val = 575
# submitting the linear regression model to output csv file
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=True, normalize=True)
x_data, y_data = shuffle(x_data, y_data)

lin_reg_model = Ridge(alpha=optimal_ridge_val)
lin_reg_model.fit(x_data, y_data['OverallScore'])
lin_reg_predictions = lin_reg_model.predict(testing_values)
lin_reg_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': lin_reg_predictions})
lin_reg_output.to_csv('lin_reg_result.csv', index=False)
# using grid search to find the optimal hyperparameters for the SVR model, C and epsilon.
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=True, normalize=True)
x_data, y_data = shuffle(x_data, y_data)

svm_model = SVR()

# The parameter grid to grid-search
params = {'C': np.arange(0.05, 3, 0.05), 'epsilon': np.arange(0,0.5, 0.005)}

# clf = GridSearchCV(svm_model, params, n_jobs=-1, cv=2, verbose=2, refit=True, return_train_score=True)
# clf.fit(x_data, y_data['OverallScore'])
# clf.best_params_
# submitting the SVR results to output csv file
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=True, normalize=True)
x_data, y_data = shuffle(x_data, y_data)

optimal_epsilon = 0.825
optimal_C = 1.9
svr_model = SVR(epsilon=optimal_epsilon, C=optimal_C)
svr_model.fit(x_data, y_data['OverallScore'])
svr_predictions = svr_model.predict(testing_values)
svr_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': svr_predictions})
svr_output.to_csv('svr_result.csv', index=False)
ensemble_predictions = (1/4)*(lin_reg_predictions+xgboost_predictions+random_forest_predictions+svr_predictions)
ensemble_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': ensemble_predictions})
ensemble_output.to_csv('ensemble_result.csv', index=False)
ensemble_predictions = (1/3)*(lin_reg_predictions+xgboost_predictions+random_forest_predictions)
ensemble_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': ensemble_predictions})
ensemble_output.to_csv('ensemble_result2.csv', index=False)
ensemble_predictions = (1/2)*(lin_reg_predictions+xgboost_predictions)
ensemble_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': ensemble_predictions})
ensemble_output.to_csv('ensemble_result3.csv', index=False)
ensemble_predictions = (1/2)*(random_forest_predictions+xgboost_predictions)
ensemble_output = pd.DataFrame({'Id': list(testing_values.index), 'OverallScore': ensemble_predictions})
ensemble_output.to_csv('ensemble_result4.csv', index=False)
# testing xgboost model
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=True, normalize=False)
x_data, y_data = shuffle(x_data, y_data)

xgboost_reg = XGBRegressor(colsample_bytree=0.8,  gamma=0.5, learning_rate=0.02, max_depth=18, min_child_weight=13, n_estimators=1000)
print(cross_val_score(xgboost_reg, x_data, y_data['ATScore'], cv=5, n_jobs=-1).mean())
# testing random forest model
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=False, normalize=False)
x_data, y_data = shuffle(x_data, y_data)

optimal_random_forest_val=220
randomForest = RandomForestRegressor(n_estimators=optimal_random_forest_val)
print(cross_val_score(randomForest, x_data, y_data['ATScore'], cv=5, n_jobs=-1).mean())
# testing linear regression model
x_data, y_data, testing_values = preprocess_all_at_once(drop_na_cols=2, categorical=2, intercept=True, normalize=True)
x_data, y_data = shuffle(x_data, y_data)

optimal_ridge_val = 575
lin_reg = Ridge(alpha=optimal_ridge_val)
print(cross_val_score(lin_reg, x_data, y_data['ATScore'], cv=5, n_jobs=-1).mean())
