import pandas as pd
import numpy as np
# Reading in the data
shoe_data = pd.read_csv('/content/Clean_Shoe_Data.csv', parse_dates = True)
df = shoe_data.copy()
# Renaming columns to get rid of spaces 
df = df.rename(columns={
    "Order Date": "Order_date",
    "Sneaker Name": "Sneaker_Name",
    "Sale Price": "Sale_Price",
    "Retail Price": "Retail_Price",
    "Release Date": "Release_Date",
    "Shoe Size": "Shoe_Size",
    "Buyer Region": "Buyer_Region"
    })
# Converting dates into numericals
import datetime as dt

df['Order_date'] = pd.to_datetime(df['Order_date'])
df['Order_date']=df['Order_date'].map(dt.datetime.toordinal)

df['Release_Date'] = pd.to_datetime(df['Release_Date'])
df['Release_Date']=df['Release_Date'].map(dt.datetime.toordinal)
# Setting up train & test data
from sklearn.model_selection import train_test_split

X = df.drop(['Sale_Price'], axis=1)
y = df.Sale_Price
X_train,X_valid,y_train,y_valid = train_test_split(X, y, test_size=0.2, random_state = 27)
# Converting categorical data to numerical
from sklearn.preprocessing import OneHotEncoder

object_cols = ['Sneaker_Name', 'Buyer_Region', 'Brand']
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# # Adding the column names after one hot encoding
OH_cols_train.columns = OH_encoder.get_feature_names(object_cols)
OH_cols_valid.columns = OH_encoder.get_feature_names(object_cols)

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
X_train.head()
X_train.shape
X_valid.head()
# Importing models
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
# Setting up pipelines

from sklearn.pipeline import Pipeline
# Decision Tree Regression Pipeline
pipeline_dtr=Pipeline([('dtr', DecisionTreeRegressor(random_state=27))])

# Random Forest Pipeline
pipeline_randomforest=Pipeline([('rf_regressor',RandomForestRegressor(random_state=27))])

# XGBost Pipeline
pipeline_xgb=Pipeline([('xgb_regressor',xgb.XGBRegressor(objective="reg:linear", random_state=27))])
# Creating a list of the pipelines to loop through them
pipelines = [pipeline_dtr, pipeline_xgb, pipeline_randomforest]

best_accuracy=0.0
best_regressor=0
best_pipeline=""

# Dictionary of pipelines and regression types for ease of reference
pipe_dict = {0: 'DTR', 1: 'XGBoost', 2: 'RandomForest'}

# Fit the pipelines
for pipe in pipelines:
	pipe.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
# Checking the accuracy of each model
for i,model in enumerate(pipelines):
    print("{} Test Accuracy: {}".format(pipe_dict[i],model.score(X_valid,y_valid)))

# Finding the best model
for i,model in enumerate(pipelines):
    if model.score(X_valid,y_valid)>best_accuracy:
        best_accuracy=model.score(X_valid,y_valid)
        best_pipeline=model
        best_regressor=i
print('Model with best accuracy: {}'.format(pipe_dict[best_regressor]))
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
# Using Randomized Search CV to find the best parameters

# Number of trees in random forest
n_estimators = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 25, 50, 75, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Method of selecting samples for training each tree
# bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

pprint(random_grid)
# THIS BLOCK TAKES EONS
rf = RandomForestRegressor(random_state=27)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=27, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)
# Evaluation of Random Search
def evaluate(model, X_valid, y_valid):
    predictions = model.predict(X_valid)
    errors = np.sqrt(mean_squared_error(y_valid, predictions))
    print('Model Performance')
    print('MSE of: ', errors)
    
    return errors
from sklearn.metrics import mean_squared_error
base_model = rf
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_valid, y_valid)


best_random = rf_random.best_estimator_
best_random.fit(X_train , y_train)

random_accuracy = evaluate(best_random, X_valid, y_valid)

print('\n')
print('Base Model Error: ', base_accuracy)
print('\n')
print('Improved Model Error: ', random_accuracy)
print('Improvement of {:0.2f}%.'.format((random_accuracy - base_accuracy) / base_accuracy))

print('\n')
print('RF_Randomized_Search_CV is complete.')
print('\n')
print('The best model is',rf_random.best_estimator_)
print('The accuracy of this model is', rf_random.best_score_*100 + '%')
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(best_random, X_valid, y_valid,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):\n")
print(scores.mean())
# Saving model to disk
import pickle
pickle.dump(best_random, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))