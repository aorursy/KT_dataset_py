import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,median_absolute_error,r2_score
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

import tensorflow as tf
from tensorflow.estimator import LinearRegressor
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()
housing.head(10)
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
housing.hist(bins = 50, figsize= (20,15))
housing = housing[housing['median_house_value']< 500001]
housing.reset_index(drop=True,inplace = True)

housing.info()
#based on median_icome data, we try to classify it into five categories
housing['income_category'] = pd.cut(housing['median_income'],bins = [0,1.5,3,4.5,6,np.inf],labels = [1,2,3,4,5])
#splitting
ss_split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.20, random_state = 42)
for train_index, test_index in ss_split.split(housing, housing['income_category']):
    train_set = housing.loc[train_index]
    test_set = housing.loc[test_index]
#dropping income_category attribute
for set in (train_set,test_set):
    set.drop('income_category',axis=1,inplace = True )
#making a copy of training data
train_set.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.1)
train_set.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.5, s= train_set['population']/100, 
             label = 'population' , figsize = (20,10) , c = 'median_house_value', cmap = plt.get_cmap('jet'), 
             colorbar= True)
corr_matrix = train_set.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)
attributes = ['median_house_value','median_income','total_rooms','housing_median_age','latitude']
scatter_matrix(train_set[attributes],figsize = (12,8))
#try out various attribute combinations
train_set['rooms_per_household'] = train_set['total_rooms']/train_set['households']
train_set['bedrooms_per_room'] = train_set['total_bedrooms']/train_set['total_rooms']
train_set['population_per_household'] = train_set['population']/train_set['households']
corr_matrix = train_set.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
#try out various attribute combinations
test_set['rooms_per_household'] = test_set['total_rooms']/test_set['households']
test_set['bedrooms_per_room'] = test_set['total_bedrooms']/test_set['total_rooms']
test_set['population_per_household'] = test_set['population']/test_set['households']

## Splitting X & y from Data sets to extract the dependent variable y away from processing data

#y_train = train_set['median_house_value'].values
#X_train = train_set.copy()
#X_train.drop('median_house_value', axis = 1, inplace = True)
#y_test = test_set['median_house_value'].values
#X_test = test_set.copy()
#X_test.drop('median_house_value', axis = 1, inplace = True)
#impute = SimpleImputer(missing_values=np.nan, strategy = 'median')
#train_num = X_train.drop('ocean_proximity', axis =1)
#impute.fit_transform(train_num)
#train_num.head(10)
#impute.statistics_
#std_scaler = StandardScaler()
#train_num_array = std_scaler.fit_transform(train_num)
#train_num = pd.DataFrame(train_num_array , columns = train_num.columns , index = train_num.index)
#train_num['ocean_proximity'] = X_train['ocean_proximity']
#X_train = train_num.copy()
#category_trans = make_column_transformer((OneHotEncoder(),['ocean_proximity']),remainder = 'passthrough')
#X_train = category_trans.fit_transform(X_train)
##X_train = pd.DataFrame(housing_array , columns = housing.columns , index = housing.index)
#X_train.shape
# Splitting X & y from Data sets to extract the dependent variable y away from processing data

y_train = train_set['median_house_value'].values
X_train = train_set.copy()
X_train.drop('median_house_value', axis = 1, inplace = True)
y_test = test_set['median_house_value'].values
X_test = test_set.copy()
X_test.drop('median_house_value', axis = 1, inplace = True)

train_num = X_train.drop('ocean_proximity', axis =1)
test_num = X_test.drop('ocean_proximity', axis =1)
X_train.shape
num_attributes = list(train_num)
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),('std_scaler', StandardScaler())])
pipeline = ColumnTransformer([('num_Pipeline', num_pipeline,num_attributes),
                                 ('category', OneHotEncoder(),cat_attributes)],remainder='passthrough')
X_train= pipeline.fit_transform(X_train)

X_train.shape                                                     
X_test= pipeline.transform(X_test)
X_test.shape
reg_model = LinearRegression()
reg_model.fit(X_train,y_train)
y_pred = reg_model.predict(X_train)

print('the training score = ',reg_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
ridge_model = Ridge()
ridge_model.fit(X_train,y_train)
y_pred = ridge_model.predict(X_train)
print('the training score = ',ridge_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train,y_train)
y_pred = tree_model.predict(X_train)
print('the training score = ',tree_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
tree_score = cross_val_score(tree_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
tree_rmse_score = np.sqrt(-tree_score)
def dispaly_scores(scores):
    print('scores : ', scores)
    print('mean = ', scores.mean())
    print('standard deviation = ',scores.std())

dispaly_scores(tree_rmse_score)
forest_model = RandomForestRegressor()
forest_model.fit(X_train,y_train)
y_pred = forest_model.predict(X_train)
print('the training score = ',forest_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
forest_score = cross_val_score(forest_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
forest_rmse_score = np.sqrt(-forest_score)
dispaly_scores(forest_rmse_score)
svr_model = SVR()
svr_model.fit(X_train,y_train)
y_pred = svr_model.predict(X_train)
print('the training score = ',svr_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
gbr_model = GradientBoostingRegressor()
gbr_model.fit(X_train,y_train)
y_pred = gbr_model.predict(X_train)
print('the training score = ',gbr_model.score(X_train,y_train))
mse = mean_squared_error(y_train,y_pred)
rmse = np.sqrt(mse)
print('the root mean squared error = ', rmse)
gbr_score = cross_val_score(gbr_model,X_train,y_train,scoring= 'neg_mean_squared_error', cv=10)
gbr_rmse_score = np.sqrt(-gbr_score)
dispaly_scores(gbr_rmse_score)
tune_pipeline = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',RandomForestRegressor(random_state = 42))])

grid_search = GridSearchCV( estimator = tune_pipeline, param_grid = {'selector__k':[14,16] , 
  'model__n_estimators':np.arange(360,370,10),'model__max_depth':[15]}, n_jobs=-1, scoring=["neg_mean_squared_error",'neg_mean_absolute_error'],refit = 'neg_mean_absolute_error', cv=5, verbose=3)

grid_search.fit(X_train,y_train)
print('the best parameters : ',grid_search.best_params_)
print('the best score = ', np.sqrt(-grid_search.best_score_))
grid_search.best_estimator_.score(X_train,y_train)
tune_pipeline_svr = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',SVR())])

grid_search_svr = GridSearchCV( estimator = tune_pipeline_svr, param_grid = {'selector__k':[14,16] , 
  'model__kernel':['linear'],'model__C':[5000,10000],'model__epsilon':[0.3,3]}, n_jobs=-1, scoring="neg_mean_squared_error", cv=5, verbose=3)
grid_search_svr.fit(X_train,y_train)
print('the best parameters : ',grid_search_svr.best_params_)
print('the best score = ', np.sqrt(-grid_search_svr.best_score_))
tune_pipeline_gbr = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',GradientBoostingRegressor(random_state=42))])

grid_search_gbr = GridSearchCV( estimator = tune_pipeline_gbr, param_grid = {'selector__k':[14,16] , 
  'model__loss':['ls'],'model__max_depth':[6,7],'model__learning_rate':[0.1,0.2],'model__n_estimators':[500]}, n_jobs=-1, scoring=["neg_mean_squared_error",'neg_mean_absolute_error'],refit = 'neg_mean_absolute_error', cv=5, verbose=3)
grid_search_gbr.fit(X_train,y_train)
print('the best parameters : ',grid_search_gbr.best_params_)
print('the best score = ', np.sqrt(-grid_search_gbr.best_score_))
tune_pipeline_ridge = Pipeline([
     ('selector',SelectKBest(f_regression)),
     ('model',Ridge(random_state=42))])

grid_search_ridge = GridSearchCV( estimator = tune_pipeline_ridge, param_grid = {'selector__k':[15,16] , 
  'model__alpha':[0.5,1]}, n_jobs=-1, scoring="neg_mean_squared_error", cv=5, verbose=3)
grid_search_ridge.fit(X_train,y_train)
print('the best parameters : ',grid_search_ridge.best_params_)
print('the best score = ', np.sqrt(-grid_search_ridge.best_score_))
grid_search_gbr.best_estimator_.score(X_train,y_train)
final_model = grid_search_gbr.best_estimator_
y_pred = final_model.predict(X_test)
final_model.score(X_test,y_test)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
mae = mean_absolute_error(y_test,y_pred)
median_ae = median_absolute_error(y_test,y_pred)
print(rmse)
print(mae)
print(median_ae)
y_train = train_set['median_house_value']
X_train = train_set.copy()
X_train.drop('median_house_value', axis = 1, inplace = True)
y_test = test_set['median_house_value']
X_test = test_set.copy()
X_test.drop('median_house_value', axis = 1, inplace = True)
X_train.info()
train_num = X_train.drop('ocean_proximity', axis =1)
test_num = X_test.drop('ocean_proximity', axis =1)
num_attributes = list(train_num)
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),('std_scaler', StandardScaler())])
processing_pipeline = ColumnTransformer([('num_Pipeline', num_pipeline,num_attributes)],remainder='passthrough')

X_train_ = processing_pipeline.fit_transform(X_train)
X_test_ = processing_pipeline.transform(X_test)
X_train = pd.DataFrame(X_train_,columns=num_attributes+['ocean_proximity'],index=X_train.index)
X_test = pd.DataFrame(X_test_,columns=num_attributes+['ocean_proximity'],index=X_test.index)
X_train[[i for i in X_train.columns if i not in ['ocean_proximity']]] = X_train[[i for i in X_train.columns if i not in ['ocean_proximity']]].apply(pd.to_numeric)
X_test[[i for i in X_test.columns if i not in ['ocean_proximity']]] = X_test[[i for i in X_test.columns if i not in ['ocean_proximity']]].apply(pd.to_numeric)
X_test.info()
feature_columns_numeric = [tf.feature_column.numeric_column(m) for m in train_num.columns]
feature_columns_categorical = [tf.feature_column.categorical_column_with_hash_bucket('ocean_proximity',
                                                                                     hash_bucket_size=1000)]
feature_columns = feature_columns_numeric + feature_columns_categorical
feature_columns
dataset = tf.data.Dataset.from_tensor_slices((dict(X_train), y_train))
dataset
def feed_input(features_dataframe, target_dataframe, num_of_epochs=10, shuffle=True, batch_size=32):
  def input_feed_function():
    dataset = tf.data.Dataset.from_tensor_slices((dict(features_dataframe), target_dataframe))
    if shuffle:
      dataset = dataset.shuffle(2000)
    dataset = dataset.batch(batch_size).repeat(num_of_epochs)
    return dataset
  return input_feed_function

train_feed_input = feed_input(X_train, y_train)
train_feed_input_testing = feed_input(X_train, y_train, num_of_epochs=1, shuffle=False)
test_feed_input = feed_input(X_test, y_test, num_of_epochs=1, shuffle=False)

regression_model = LinearRegressor(feature_columns)
regression_model.train(train_feed_input,steps=1000)
train_predictions = regression_model.predict(train_feed_input_testing)
test_predictions = regression_model.predict(test_feed_input)
train_predictions_series = pd.Series([p['predictions'][0] for p in train_predictions])
test_predictions_series = pd.Series([p['predictions'][0] for p in test_predictions])
train_predictions_df = pd.DataFrame(train_predictions_series, columns=['predictions'])
test_predictions_df = pd.DataFrame(test_predictions_series, columns=['predictions'])
y_train.reset_index(drop=True, inplace=True)
train_predictions_df.reset_index(drop=True, inplace=True)

y_test.reset_index(drop=True, inplace=True)
test_predictions_df.reset_index(drop=True, inplace=True)
train_labels_with_predictions_df = pd.concat([y_train, train_predictions_df], axis=1)
test_labels_with_predictions_df = pd.concat([y_test, test_predictions_df], axis=1)
def calculate_errors_and_r2(y_true, y_pred):
  mean_squared_err = (mean_squared_error(y_true, y_pred))
  root_mean_squared_err = np.sqrt(mean_squared_err)
  r2 = round(r2_score(y_true, y_pred)*100,0)
  return mean_squared_err, root_mean_squared_err, r2
train_mean_squared_error, train_root_mean_squared_error, train_r2_score_percentage = calculate_errors_and_r2(y_train, train_predictions_series)
test_mean_squared_error, test_root_mean_squared_error, test_r2_score_percentage = calculate_errors_and_r2(y_test, test_predictions_series)

print('Training Data Mean Squared Error = ', train_mean_squared_error)
print('Training Data Root Mean Squared Error = ', train_root_mean_squared_error)
print('Training Data R2 = ', train_r2_score_percentage)

print('Test Data Mean Squared Error = ', test_mean_squared_error)
print('Test Data Root Mean Squared Error = ', test_root_mean_squared_error)
print('Test Data R2 = ', test_r2_score_percentage)