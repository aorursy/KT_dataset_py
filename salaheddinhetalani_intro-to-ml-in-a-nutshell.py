# # # # # # # IMPORTS # # # # # # #



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error





# # # # # # # GETTING DATA # # # # # # #



home_data_file_path = '../input/home-data-for-ml-course/train.csv' # save the path of the file to read 

home_data = pd.read_csv(home_data_file_path, index_col='Id') # read the data and store it in DataFrame titled home_data





# # # # # # # EXPLORING DATA # # # # # # #



obj_cols = [col for col in home_data.columns if home_data[col].dtype == 'object'] 

num_cols = [col for col in home_data.columns if home_data[col].dtype != 'object'] 

missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()] 



print("Home Data | #rows: {}, #columns: {} (#numerical: {}, #categorical: {}), #columns with missing values: {}\n".format(home_data.shape[0], home_data.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))





# # # # # # # DATA MANIPULATION # # # # # # #



# SELECT TARGET #

y = home_data.SalePrice # we use the dot notation to select the column we want to predict 



# SELECT FEATURES #

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'] # to keep things simple, we choose only 7 features out of 80 available, all numerical

X = home_data[features] 



# SPLIT DATA INTO TRAINING AND VALIDAION SETS #

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0) # supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script



print("Training Data | #rows: {}, #features: {}".format(X_train.shape[0], X_train.shape[1]))

print("Validation Data | #rows: {}, #features: {}\n".format(X_valid.shape[0], X_valid.shape[1]))





# # # # # # # DECESION TREE MODEL # # # # # # #



# DEFINE #

decision_tree_model = DecisionTreeRegressor(random_state=0) # many machine learning models allow some randomness in model training. Specifying a number for random_state ensures you get the same results in each run.

# FIT #

decision_tree_model.fit(X_train, y_train) # fit the Model

# PREDICT #

val_predictions = decision_tree_model.predict(X_valid) # make validation predictions

# EVALUATE

val_mae = mean_absolute_error(val_predictions, y_valid) # calculate mean absolute error



print("Validation MAE | Decision Tree Model (when not specifying max_leaf_nodes) -> {:,.0f}".format(val_mae))





# Define a function to calculate MAE scores from different values for max_leaf_nodes

def get_mae(max_leaf_nodes):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, val_predictions)



candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

scores = {leaf_size: get_mae(leaf_size) for leaf_size in candidate_max_leaf_nodes}

best_tree_size = min(scores, key=scores.get)



decision_tree_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1) 

decision_tree_model.fit(X_train, y_train)

val_predictions = decision_tree_model.predict(X_valid)

val_mae = mean_absolute_error(val_predictions, y_valid)



print("Validation MAE | Decision Tree Model (when best value of max_leaf_nodes '{}' is chosen) -> {:,.0f}".format(best_tree_size, val_mae) + "\n")





# # # # # # # RANDOM FOREST MODEL # # # # # # #



random_forest_model = RandomForestRegressor(random_state=0)

random_forest_model.fit(X_train, y_train)

val_predictions = random_forest_model.predict(X_valid)

val_mae = mean_absolute_error(val_predictions, y_valid)



print("Validation MAE | Random Forest Model -> {:,.0f}".format(val_mae))





model_1 = RandomForestRegressor(n_estimators=50, random_state=0)

model_2 = RandomForestRegressor(n_estimators=100, random_state=0)

model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)

model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

model_6 = RandomForestRegressor(n_estimators=300, criterion='mae', random_state=0)

model_7 = RandomForestRegressor(n_estimators=150, min_samples_split=10, criterion='mae', random_state=0)



models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7]



# Function for comparing different models

def score_model(model):

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, val_predictions)



for i in range(0, len(models)):

    val_mae = score_model(models[i])

    print("Validation MAE | Random Forest Model {} -> {:,.0f}".format(i+1, val_mae))
# # # # # # # IMPORTS # # # # # # #



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error





# # # # # # # GETTING DATA # # # # # # #



home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')





# # # # # # # DATA MANIPULATION # # # # # # #



home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset



# SELECT TARGET #

y = home_data.SalePrice



# SELECT FEATURES #

home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset

X = home_data.select_dtypes(exclude=['object']) # to keep things simple, we'll use only numerical predictors



# SPLIT DATA INTO TRAINING AND VALIDAION SETS #

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)





# # # # # # # EXPLORING TRAINING DATA # # # # # # #



obj_cols = [col for col in X_train.columns if X_train[col].dtype == 'object'] 

num_cols = [col for col in X_train.columns if X_train[col].dtype != 'object'] 

missing_val_cols = [col for col in X_train.columns if X_train[col].isnull().any()] 



print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))



missing_val_count_by_column = X_train.isnull().sum()



print(missing_val_count_by_column[missing_val_count_by_column > 0])



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, val_predictions)





# # # # # # # DROP COLUMNS WITH MISSING VALUES # # # # # # #



reduced_X_train = X_train.drop(missing_val_cols, axis=1)

reduced_X_valid = X_valid.drop(missing_val_cols, axis=1)



print("\nValidation MAE | Drop columns with missing values -> {:,.0f}".format(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)))





# # # # # # # IMPUTATION # # # # # # #



# Impute missing values with the mean value along each column

imputer = SimpleImputer(strategy='mean') 

imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns



print("Validation MAE | Imputation -> {:,.0f}".format(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)))





# # # # # # # IMPUTATION PLUS # # # # # # #



# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()



# Make new columns indicating what will be imputed

for col in missing_val_cols:

    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    

my_imputer = SimpleImputer(strategy='mean')

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_valid_plus.columns = X_valid_plus.columns



print("Validation MAE | Imputation Plus -> {:,.0f}".format(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)))
# # # # # # # IMPORTS # # # # # # #



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error





# # # # # # # GETTING DATA # # # # # # #



home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')





# # # # # # # DATA MANIPULATION # # # # # # #



home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset



# SELECT TARGET #

y = home_data.SalePrice



# SELECT FEATURES #

home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset



missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()] # get names of columns with missing values 

X = home_data.drop(missing_val_cols, axis=1) # to keep things simple, we'll drop columns with missing values



# SPLIT DATA INTO TRAINING AND VALIDAION SETS #

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)





# # # # # # # EXPLORING TRAINING DATA # # # # # # #



obj_cols = [col for col in X_train.columns if X_train[col].dtype == 'object'] 

num_cols = [col for col in X_train.columns if X_train[col].dtype != 'object'] 

missing_val_cols = [col for col in X_train.columns if X_train[col].isnull().any()] 



print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, val_predictions)





# # # # # # # DROP CATEGORICAL VARIABLES # # # # # # #



drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_valid = X_valid.select_dtypes(exclude=['object'])



print("Validation MAE | Drop categorical variables -> {:,.0f}".format(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)))





# # # # # # # LABEL ENCODING # # # # # # #



# CAUTION: Values might differ between the training and validation set, thus resulting an error when fitting different values in the validation set

print("\nUnique values in 'Condition2' column in training data:", X_train['Condition2'].unique())

print("Unique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())



# Columns that can be safely label encoded

good_label_cols = [col for col in obj_cols if set(X_train[col]) == set(X_valid[col])]



# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(obj_cols)-set(good_label_cols))



print("\n#categorical features that will be label encoded: {}, #categorical features that will be dropped: {}\n".format(len(good_label_cols), len(bad_label_cols)))



# Drop categorical columns that will not be encoded

label_X_train = X_train.drop(bad_label_cols, axis=1)

label_X_valid = X_valid.drop(bad_label_cols, axis=1)



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder() 

for col in good_label_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])   

    

print("Validation MAE | Label Encoding -> {:,.0f}\n".format(score_dataset(label_X_train, label_X_valid, y_train, y_valid)))





# # # # # # # ONE-HOT ENCODING # # # # # # #



# Columns that will be one-hot encoded

low_cardinality_cols = [col for col in obj_cols if X_train[col].nunique() < 10]



# Columns that will be dropped from the dataset

high_cardinality_cols = list(set(obj_cols)-set(low_cardinality_cols))



print("#categorical features that will be one-hot encoded: {}, #categorical features that will be dropped: {}\n".format(len(low_cardinality_cols), len(high_cardinality_cols)))



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # handle_unknown='ignore' -> to avoid errors when the validation data contains classes that aren't represented in the training data; sparse=False -> ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols])) 

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols])) 



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Remove categorical columns (as they will be replaced with one-hot encoding)

num_X_train = X_train.drop(obj_cols, axis=1)

num_X_valid = X_valid.drop(obj_cols, axis=1)



# Add the one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



print("Validation MAE | One-Hot Encoding -> {:,.0f}".format(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)))
# # # # # # # IMPORTS # # # # # # #



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline



# # # # # # # GETTING DATA # # # # # # #



home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')





# # # # # # # DATA MANIPULATION # # # # # # #



home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset



# SELECT TARGET #

y = home_data.SalePrice



# SELECT FEATURES #

home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset



obj_cols = [col for col in home_data.columns if home_data[col].nunique() < 10 and home_data[col].dtype == "object"]

num_cols = [col for col in home_data.columns if home_data[col].dtype in ['int64', 'float64']]

missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()] 



selected_cols = obj_cols + num_cols

X = home_data[selected_cols].copy()



# SPLIT DATA INTO TRAINING AND VALIDAION SETS #

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)





# # # # # # # EXPLORING TRAINING DATA # # # # # # #



print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))





# # # # # # # DEFINE PREPROSSING STEPS # # # # # # #



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(

    steps=[('imputer', SimpleImputer(strategy='most_frequent')), 

           ('onehot', OneHotEncoder(handle_unknown='ignore'))])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[('num', numerical_transformer, num_cols), 

                  ('cat', categorical_transformer, obj_cols)])





# # # # # # # DEFINE MODEL # # # # # # #



model = RandomForestRegressor(n_estimators=100, random_state=0)





# # # # # # # CREATE AND EVALUATE THE PIPELINE # # # # # # #



# CREATE

# Bundle preprocessing and modeling code in a pipeline

pipeline = Pipeline( 

    steps=[('preprocessor', preprocessor), 

           ('model', model)]) 

# FIT

pipeline.fit(X_train, y_train) # preprocessing of training data, fit model 

# PREDICT 

val_predictions = pipeline.predict(X_valid) # preprocessing of validation data, get predictions

# EVALUATE

val_mae = mean_absolute_error(y_valid, val_predictions)



print("Validation MAE | Pipeline -> {:,.0f}".format(val_mae))
# # # # # # # IMPORTS # # # # # # #



import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score





# # # # # # # GETTING DATA # # # # # # #



home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')





# # # # # # # DATA MANIPULATION # # # # # # #



home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset



# SELECT TARGET #

y = home_data.SalePrice



# SELECT FEATURES #

home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset

X = home_data.select_dtypes(exclude=['object']) # to keep things simple, we'll use only numerical predictors





# # # # # # # EXPLORING TRAINING DATA # # # # # # #



obj_cols = [col for col in X.columns if X[col].dtype == 'object'] 

num_cols = [col for col in X.columns if X[col].dtype != 'object'] 

missing_val_cols = [col for col in X.columns if X[col].isnull().any()] 



print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X.shape[0], X.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))





# # # # # # # PREPROSESSING & MODELING # # # # # # #



pipeline = Pipeline(

    steps=[('preprocessor', SimpleImputer()), 

           ('model', RandomForestRegressor(n_estimators=50, random_state=0))])





# # # # # # # CROSS-VALIDATION # # # # # # #



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error') # cv -> set the number of folds; scoring -> choose a measure of model quality to report



print("Validation MAE | Cross-Validation scores over 3 folds -> {}, average score -> {:,.0f}".format(scores, scores.mean()))





# Define a function to calculate MAE scores from different values for n_estimators

def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -> the number of trees in the forest

    """

    pipeline = Pipeline(

        steps=[('preprocessor', SimpleImputer()), 

               ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])

    scores = -1 * cross_val_score(pipeline, X, y, cv=3, scoring='neg_mean_absolute_error')

    return scores.mean()



candidate_n_estimators = [50, 100, 150, 200, 250, 300]

results = {est: get_score(est) for est in candidate_n_estimators} 

best_tree_number = min(results, key=results.get)



print("Validation MAE | Cross-Validation average score over 3 folds when best value of n_estimators ({}) is chosen -> {:,.0f}".format(best_tree_number, results[best_tree_number]))
# # # # # # # IMPORTS # # # # # # #



import pandas as pd

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error





# # # # # # # GETTING DATA # # # # # # #



home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')





# # # # # # # DATA MANIPULATION # # # # # # #



home_data.dropna(axis=0, subset=['SalePrice'], inplace=True) # remove rows with missing target from the dataset



# SELECT TARGET #

y = home_data.SalePrice



# SELECT FEATURES #

home_data.drop(['SalePrice'], axis=1, inplace=True) # drop the target column from the dataset



missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()]

home_data.drop(missing_val_cols, axis=1, inplace=True) # to keep things simple, we'll drop columns with missing values



obj_cols = [col for col in home_data.columns if home_data[col].nunique() < 10 and home_data[col].dtype == "object"]

num_cols = [col for col in home_data.columns if home_data[col].dtype in ['int64', 'float64']]

missing_val_cols = [col for col in home_data.columns if home_data[col].isnull().any()]



selected_cols = obj_cols + num_cols

X = home_data[selected_cols].copy()



# SPLIT DATA INTO TRAINING AND VALIDAION SETS #

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)





# # # # # # # EXPLORING TRAINING DATA # # # # # # #



print("Training Data | #rows: {}, #features: {} (#numerical: {}, #categorical: {}), #features with missing values: {}\n".format(X_train.shape[0], X_train.shape[1], len(num_cols), len(obj_cols), len(missing_val_cols)))





# # # # # # # ONE-HOT ENCODING # # # # # # #



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # handle_unknown='ignore' -> to avoid errors when the validation data contains classes that aren't represented in the training data; sparse=False -> ensures that the encoded columns are returned as a numpy array (instead of a sparse matrix)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[obj_cols])) 

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[obj_cols])) 



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Remove categorical columns (as they will be replaced with one-hot encoding)

num_X_train = X_train.drop(obj_cols, axis=1)

num_X_valid = X_valid.drop(obj_cols, axis=1)



# Add the one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)





# # # # # # # XGBOOST MODEL # # # # # # #



# DEFINE

xgboost_model = XGBRegressor(n_estimators=300, learning_rate=0.2) 

# FIT

xgboost_model.fit(OH_X_train, y_train, early_stopping_rounds=5, eval_set=[(OH_X_valid, y_valid)], verbose=False)

# PREDICT

val_predictions = xgboost_model.predict(OH_X_valid) 

# EVALUATE

val_mae = mean_absolute_error(val_predictions, y_valid) 



print("Validation MAE | XGBoost Model -> {:,.0f}".format(val_mae) + "\n")





model_1 = XGBRegressor(n_estimators=50, learning_rate=0.1)

model_2 = XGBRegressor(n_estimators=100, learning_rate=0.1)

model_3 = XGBRegressor(n_estimators=150, learning_rate=0.1)

model_4 = XGBRegressor(n_estimators=200, learning_rate=0.05)

model_5 = XGBRegressor(n_estimators=200, learning_rate=0.1)

model_6 = XGBRegressor(n_estimators=200, learning_rate=0.2)

model_7 = XGBRegressor(n_estimators=300, learning_rate=0.1)

model_8 = XGBRegressor(n_estimators=350, learning_rate=0.1)



models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8]



# Function for comparing different models

def score_model(model, X_t=OH_X_train, X_v=OH_X_valid, y_t=y_train, y_v=y_valid):

    model.fit(X_t, y_t)

    preds = model.predict(X_v)

    return mean_absolute_error(y_v, preds)



for i in range(0, len(models)):

    val_mae = score_model(models[i])

    print("Validation MAE | XGBoost Model {} -> {:,.0f}".format(i+1, val_mae))