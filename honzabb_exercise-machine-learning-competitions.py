# Code you have previously used to load data
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer

def get_rf_mae(train_X, val_X, train_y, val_y):
    rf_model = RandomForestRegressor(random_state=1, n_estimators=10)
    rf_model.fit(train_X, train_y)
    predicted_values = rf_model.predict(val_X)
    return mean_absolute_error(val_y,predicted_values)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
#test_data.drop('Id', axis='columns', inplace=True)
    


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
#home_data.drop('Id', axis='columns', inplace=True)
#test_data.drop('Id', axis='columns', inplace=True)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]
X_rf_features = home_data.iloc[:, [3,15,13,11,8,2,12,18,25,24]]

# Get also all columns, except for target
X_full = home_data.drop('SalePrice', axis='columns') 

# From all columns, get only numerical ones
X_num_cols = X_full.select_dtypes(exclude='object')
print('Shape of DF with num cols: {}'.format(X_num_cols.shape))

# Drop columns with missing values
X_nan_dropped = X_num_cols.dropna(axis=1)
print('Shape of DF with dropped cols with missing values: {}'.format(X_nan_dropped.shape))
# or alternative
#X_nan_dropped = home_data.drop(cols_with_missing, axis=1)

# Get columns where are some missing values
cols_with_missing = [col for col in X_num_cols.columns
                    if X_num_cols[col].isnull().any()]

# Extra columns marking columns with missing values
# For every columns with missing values, each where was the missing value is marked by true
# rows where wasn't missing value are false
X_num_cols_copy = X_num_cols.copy()
for col in cols_with_missing:
    X_num_cols_copy[col + '_was_missing'] = X_num_cols_copy[col].isnull()
    
# Handle missing values in numerical columns, with mean imputer
imputer_mean = SimpleImputer(strategy='mean')
X_num_cols_imputed = pd.DataFrame(imputer_mean.fit_transform(X_num_cols))

X_num_cols_copy_imputed = pd.DataFrame(imputer_mean.fit_transform(X_num_cols_copy))
print('Shape of DF with num cols and marked cols with missing values: {}'.format(X_num_cols_copy_imputed.shape))
#X_num_cols_copy_imputed.columns = X_num_cols.columns

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

train_X_dropped_nan, val_X_dropped_nan, train_y_dropped_nan, val_y_dropped_nan = train_test_split(X_nan_dropped, y, random_state=1)

train_num_X, val_num_X, train_num_y, val_num_y = train_test_split(X_num_cols_imputed, y, random_state=1)

train_num_X_plus, val_num_X_plus, train_num_y_plus, val_num_y_plus = train_test_split(X_num_cols_copy_imputed, y, random_state=1)

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
rf_model = RandomForestRegressor(random_state=1, n_estimators=10)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# From categorical columns, select only low cardinality ones
low_cardinality_cols_a = [col for col in home_data.select_dtypes(include='object').columns
                                       if home_data[col].nunique() < 10]

low_cardinality_cols_b = [col for col in home_data.select_dtypes(include='object').columns
                                       if home_data[col].nunique() < 8]

# Did intersection of these two lists of columns, for better set of features
low_cardinality_cols = list(set(low_cardinality_cols_a).intersection(low_cardinality_cols_b))

# Select only numeric columns
num_cols = list(home_data.select_dtypes(exclude='object').columns.values)

# Concate low cardinality categorical columns and all numerical columns
ohe_columns = low_cardinality_cols + num_cols

# Drop target column
X_ohe =  pd.get_dummies(home_data[ohe_columns].drop('SalePrice', axis='columns'))
X_to_pipeline = X_ohe
# Impute missing values
X_ohe = pd.DataFrame(imputer_mean.fit_transform(X_ohe))

X_rf_features_ohe = pd.get_dummies(X_rf_features)
X_rf_features_ohe = pd.DataFrame(imputer_mean.fit_transform(X_rf_features_ohe))

# Split into validation and training data
train_X_ohe, val_X_ohe, train_y_ohe, val_y_ohe = train_test_split(X_ohe, y, random_state=1)
train_X_ohe_fs, val_X_ohe_fs, train_y_ohe_fs, val_y_ohe_fs = train_test_split(X_rf_features_ohe, y, random_state=1)

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import SCORERS

my_pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=1, n_estimators=10))

train_X_pipe, val_X_pipe, train_y_pipe, val_y_pipe = train_test_split(X_to_pipeline, y)
my_pipeline.fit(train_X_pipe, train_y_pipe)
predictions = my_pipeline.predict(val_X_pipe)

#
X_pipe = home_data.drop('SalePrice', axis='columns') 
cat_features =  list(X_pipe.select_dtypes(include='object').columns.values)
num_features = list(X_pipe.select_dtypes(exclude='object').columns.values)

print('Numerical features count: {}\nCategorical features count: {}\n'.format(len(num_features), len(cat_features)))


num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('one_hot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', SimpleImputer(), num_features),
    ('cat', cat_transformer, cat_features)
])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestRegressor(random_state=1))
])

parameters = {'classifier__n_estimators' : [10,50,100]}

train_X_p, val_X_p, train_y_p, val_y_p = train_test_split(X_pipe, y)

CV = GridSearchCV(clf, parameters, scoring='neg_mean_absolute_error', n_jobs=1)
CV.fit(train_X_p, train_y_p)

print('Best score and parameter combination = ')

print(-CV.best_score_)    
print(CV.best_params_) 

clf.fit(train_X_p, train_y_p)
clf_predicted = clf.predict(val_X_p)
print('MAE: {}'.format(mean_absolute_error(val_y_p, clf_predicted)))
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, X_pipe, y, scoring='neg_mean_absolute_error')

print(scores)
print('Average MAE: {}'.format(-scores.mean()))
from xgboost import XGBRegressor

new_data = home_data.drop('SalePrice', axis='columns')

one_hot_encoded_training_predictors = pd.get_dummies(new_data)
one_hot_encoded_test_predictors = pd.get_dummies(test_data)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
#cols_with_missing_train = [col for col in new_data.columns 
#                                 if new_data[col].isnull().any()]

#cols_with_missing_test = [col for col in test_data.columns 
#                                 if test_data[col].isnull().any()]

#cols_with_missing = set().union(cols_with_missing_train, cols_with_missing_test)

#cat_cols_train = new_data.select_dtypes(include='object')
#cat_cols_test = test_data.select_dtypes(include='object')


#for col in cols_with_missing:
#    final_train[col+'_was_missing']=final_train[col].isnull()
#    final_test[col+'_was_missing']=final_test[col].isnull()
    
#imputer
candidate_train_predictors = pd.DataFrame(imputer_mean.fit_transform(final_train))
candidate_test_predictors = pd.DataFrame(imputer_mean.transform(final_test))
print(candidate_train_predictors.columns.values)
print(candidate_test_predictors.columns.values)

#candidate_train_fselection = SelectKBest(chi2, k=280).fit_transform(candidate_train_predictors, y)
train_X_fs, val_X_fs, train_y_fs, val_y_fs = train_test_split(candidate_train_predictors, y, random_state=1)
#candidate_test_predictors = candidate_test_predictors[candidate_train_fselection]
#xgb_model_final = XGBRegressor(n_estimators=412, learning_rate=0.05)
#xgb_model_final.fit(train_X_fs, train_y_fs, verbose=False)

# make predictions which we will submit.
#xgb_model_final_predictions = xgb_model.predict(val_X_fs)

#print("MAE: {}".format(mean_absolute_error(val_y_fs, xgb_model_final_predictions)))

#print(candidate_train_predictors.shape, candidate_test_predictors.shape)

#'ssssssss'



#new_data = home_data.drop('SalePrice', axis='columns')

#cat_cols = new_data.select_dtypes(include='object')

# From categorical columns, get only the ones with some missing values
#cols_with_missing = [col for col in cat_cols.columns
#                    if cat_cols[col].isnull().any()]
#print('Train')
#print(new_data[new_data.columns.difference(cat_cols.columns)].shape)

#new_data[new_data.columns.difference(cat_cols.columns)]
#imputer_most_frequent = 
#new_data_num_imputed = pd.DataFrame(imputer_mean.fit_transform(new_data[new_data.columns.difference(cat_cols.columns)]))
#new_data_cat_imputed = pd.DataFrame(imputer_mean.fit_transform(new_data[new_data.columns.difference(cat_cols.columns)]))
#print(new_data_num_imputed.shape)
#print(len(cols_with_missing))
#print(test_data.shape)
#print(new_data.shape)

#ohe_new_data = pd.get_dummies(new_data[cols_with_missing], prefix ='DUMMY_')
#df_train = pd.concat([new_data, ohe_new_data], axis=1)

#print(df_train.shape)
#cat_cols_test = test_data.select_dtypes(include='object')
#cols_with_missing_test = [cols for cols in cat_cols_test.columns
#                    if cat_cols_test[cols].isnull().any()]
#print('Test')
#print(cat_cols_test.shape)
#print(len(cols_with_missing_test))

#test_data_imputed = pd.DataFrame(imputer_mean.fit_transform(test_data))

# Drop rows with missing values
#print(test_data_imputed.shape)
#test_data_im_nona = test_data_imputed.dropna()
#print(test_data_imputed.shape)


#ohe_test_data = pd.get_dummies(test_data[cols_with_missing], prefix ='DUMMY_')
#df_test = pd.concat([test_data, ohe_test_data], axis=1)

#print(df_test.shape)

#final_train, final_test = df_train.align(df_test,join='left',axis=1)

#X_to_select = pd.get_dummies(new_data, )
#X_to_select = pd.DataFrame(imputer_mean.fit_transform(X_to_select))
#X_feature_selection = SelectKBest(chi2, k=280).fit_transform(X_to_select, y)
#train_X_fs, val_X_fs, train_y_fs, val_y_fs = train_test_split(X_feature_selection, y, random_state=1)
from sklearn.decomposition import PCA

pca = PCA(100)
pca.fit(X_to_select)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
num_components = 30
pca = PCA(num_components, random_state=1)
pca.fit(X_to_select)
pca_components = pca.transform(X_to_select)
train_X_pca, val_X_pca, train_y_pca, val_y_pca = train_test_split(pca_components, y, random_state=1)
# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=1, n_estimators=10)

# Fit model
rf_model_on_full_data.fit(X_num_cols_imputed,y)

rf_model_on_full_data_ohe = RandomForestRegressor(random_state=1, n_estimators=10)
rf_model_on_full_data_ohe.fit(X_ohe, y)


# Plot top 40 features by random forest importance
rf_model_on_full_data_ohe = pd.Series(rf_model_on_full_data_ohe.feature_importances_, index=X_ohe.columns).sort_values(ascending=False)
rf_model_on_full_data_ohe[:40].plot(kind='bar', title='Feature Importance with Random Forest', figsize=(12,8))
plt.ylabel('Feature Importance values')
plt.subplots_adjust(bottom=0.25)
plt.show()
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
xgb_model.fit(train_X_fs, train_y_fs, verbose=False)
xgb_predictions = xgb_model.predict(val_X_fs)
print('MAE, XGB untuned: {}'.format(mean_absolute_error(val_y_fs, xgb_predictions)))
# Lower learning rate => higher accuracy, and longer training
# early_stopping_rounds => can prevent overfitting
# n_jobs => paralelism
xgb_model_tuned = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs = 4)
xgb_model_tuned.fit(train_X_fs, train_y_fs, early_stopping_rounds=10, 
             eval_set=[(val_X_fs, val_y_fs)], verbose=False)
xgb_predictions = xgb_model_tuned.predict(val_X_fs)
print('MAE, XGB best iteration {}: {}'.format(xgb_model_tuned.best_iteration, mean_absolute_error(val_y_fs, xgb_predictions)))
XGB_best_iteration = xgb_model_tuned.best_iteration
xgb_model = XGBRegressor(n_estimators=XGB_best_iteration, learning_rate=0.05)
xgb_model.fit(train_X_fs, train_y_fs, verbose=False)
xgb_predictions = xgb_model.predict(val_X_fs)
print('MAE, XGB tuned: {}'.format(mean_absolute_error(val_y_fs, xgb_predictions)))
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier


gbr = GradientBoostingRegressor()
gbr.fit(X, y)
my_plots = plot_partial_dependence(gbr, 
                                   features=[0,1], 
                                   X=X, 
                                   feature_names=['Lot Area', 'Year Built'], 
                                   grid_resolution=10)

#target_feature = (0,1)
#partial_dependence(gbr, target_feature, X=X, grid_resolution=10)
#partial_dependence?
from xgboost import plot_importance

plot_importance(xgb_model, max_num_features=10);


# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

xgb_model_final = XGBRegressor(n_estimators=XGB_best_iteration, learning_rate=0.05)
xgb_model_final.fit(candidate_train_predictors, y, verbose=False)

# make predictions which we will submit.
xgb_model_final_predictions = xgb_model.predict(candidate_test_predictors)
#test_preds = rf_model.predict(test_X)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': xgb_model_final_predictions})

output.to_csv('submission.csv', index=False)

print('MAE, dropped missing columns: {}'.format(get_rf_mae(train_X_dropped_nan,val_X_dropped_nan, train_y_dropped_nan, val_y_dropped_nan)))
print('MAE, selected features: {}'.format(get_rf_mae(train_X,val_X, train_y, val_y)))
print('MAE, num features, mean imputer: {}'.format(get_rf_mae(train_num_X, val_num_X, train_num_y, val_num_y)))
print('MAE, num features, mean imputer, marked missing values: {}'.format(get_rf_mae(train_num_X_plus, val_num_X_plus, train_num_y_plus, val_num_y_plus)))
print('MAE, One Hot Encoding: {}'.format(get_rf_mae(train_X_ohe,val_X_ohe, train_y_ohe, val_y_ohe)))
#print('MAE, One Hot Encoding, SelectKBest: {}'.format(get_rf_mae(train_X_fs,val_X_fs, train_y_fs, val_y_fs)))
print('MAE, One Hot Encoding, PCA {} components: {}'.format(num_components, get_rf_mae(train_X_pca,val_X_pca, train_y_pca, val_y_pca)))
print('MAE, One Hot Encoding, TOP 10 Features by RF importance: {}'.format(get_rf_mae(train_X_ohe_fs,val_X_ohe_fs, train_y_ohe_fs, val_y_ohe_fs)))