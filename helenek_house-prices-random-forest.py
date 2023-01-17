import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error
from category_encoders import TargetEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import house_prices_utils

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# import data

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df['SalePrice'] = np.log1p(df['SalePrice'])
df_labels = df['SalePrice'].copy()

# split into train and validation set before doing target mean encoding to prevent leakage! 

train_set, val_set, train_lab, val_lab = train_test_split(df, df_labels, random_state = 42)
df.describe()
train_set.describe()
val_set.describe()
train_set.sample(10)
#encode the training set using target mean encoding for categorical values (function in utils.py)

train_set = encode_dataframe(train_set, train=True)
#train_set.sample(10)
# Problem: some categorical feature values appear in the validation set but not in the training set, so they
# do not have a target mean that can be mapped onto the feature values in the validation set

# setp 1: create dictionary with global SalePrice mean for every categorical feature value in the whole dataframe

train_val_categorical_global_means = {feature_encoded: 
 {feature_value: 
  df["SalePrice"].mean() for feature_value in df[feature].unique()} 
 for feature, feature_encoded in zip(categorical_features, categorical_features_encoded)}

# feature_encoded: {...} --> create dictionary key for all encoded categorical features as name (e.g. 'Neighborhood_encoded')
# feature_value: ... --> create dictionary in dictionary with feature_value as key for each categorical feature (e.g. 'Veenker')
# df.['SalePrice'].mean() --> create value for each feature_value (value is always mean of SalePrice)
# result: {'Neighborhood_encoded': {'Veenker': 180921.19589041095}}

train_val_categorical_global_means
# step 2: create dictionary with encoded mean (as calculated in train_set) for every categorical feature value 
# (does only include (encoded) features of the training set, not the validation set)

train_categorical_encoded_means = {}

for feature, feature_encoded in zip(categorical_features, categorical_features_encoded):
    # get encoded mean of every categorical feature and store it in mean_by_var
    mean_by_var = train_set.groupby(feature)[feature_encoded].mean().to_dict()
    # store mean_by_var as the value for feature_encoded as key in a dictionary
    train_categorical_encoded_means[feature_encoded] = mean_by_var
    
train_categorical_encoded_means
# step 3: update the first dictionary (train_val_categorical_global_means) with encoded means instead of the global
# means, so that the feature values that appear in the training set (+ validation set) are encoded means and the feature
# values that appear only in the validation set remain the global mean

# loop through keys (encoded_feature) in dict train_val_categorical_global_means
for encoded_feature in train_val_categorical_global_means.keys():
    # update the value of each key (encoded_feature) with the value of each key in train_categorical_encoded_means
    # https://www.geeksforgeeks.org/python-dictionary-update-method/
    train_val_categorical_global_means[encoded_feature].update(train_categorical_encoded_means[encoded_feature])
    
train_val_categorical_global_means
# this dictionary can now be used to encode the validation set with the encode_dataframe() function below
val_set = encode_dataframe(val_set, train=False, 
                           train_set_categorical_encoded_means=train_val_categorical_global_means)
#val_set.sample(10)
train_set_selected_features = train_set[categorical_features_encoded + ordinal_features_encoded + numerical_features]
val_set_selected_features = val_set[categorical_features_encoded + ordinal_features_encoded + numerical_features]
#train_set_selected_features.sample(10)
#val_set_selected_features.sample(10)
# convert all columns to numeric

train_set_selected_features = train_set_selected_features.apply(pd.to_numeric)
val_set_selected_features = val_set_selected_features.apply(pd.to_numeric)
train_set_selected_features.sample(5)
val_set_selected_features.sample(5)
max_depth = [3, 5, 10, 25, 35, 50]
max_features = ['auto', 'sqrt', 'log2', 3, 5, 10, 15]
min_samples_leaf = [2, 5, 10, 15, 20]
min_samples_split = [2, 5, 10, 15]
n_estimators = [100, 200, 300, 500, 800, 1000]
param_grid = {
    'max_depth': max_depth,
    'max_features': max_features,
    'min_samples_leaf': min_samples_leaf,
    'min_samples_split': min_samples_split,
    'n_estimators': n_estimators
}
from sklearn.model_selection import RandomizedSearchCV
rf = ExtraTreesRegressor()
rf_random = RandomizedSearchCV(estimator = rf,
                              param_distributions = param_grid,
                              
                              #number of random parameter combinations that are tried out
                              n_iter = 100,
                              cv = 5,
                              
                              #print messages while running
                              verbose = 2,
                              
                              #use all available processors
                              n_jobs = -1)
rf_random.fit(train_set_selected_features, train_lab)
rf_random.best_params_
best_grid_random_search = rf_random.best_estimator_
param_grid = {
    'bootstrap': [True],
    'max_depth': [40, 50, 60],
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [900, 1000, 1100]
}
#model = ExtraTreesRegressor() 
model = RandomForestRegressor()
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3)
grid_search.fit(train_set_selected_features, train_lab)
grid_search.best_params_
best_grid = grid_search.best_estimator_
model = RandomForestRegressor(bootstrap = True, max_depth = 40, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 4, n_estimators = 900 , n_jobs = -1)
model.fit(train_set_selected_features, train_lab)
sale_price_predicted = model.predict(val_set_selected_features)
sale_price_predicted_exp = np.expm1(sale_price_predicted)
sale_price_predicted_exp[:10]
val_lab_exp = np.expm1(val_lab)
val_lab_exp[:10]
mean_squared_error(val_lab, sale_price_predicted)**0.5
preds = np.stack(tree.predict(val_set_selected_features) for tree in model.estimators_)
plt.plot([r2_score(val_lab, np.mean(preds[:i+1], axis = 0)) for i in range(200)]);
importance = sorted(model.feature_importances_, reverse = True)
np.argsort(model.feature_importances_)
features = numerical_features + categorical_features_encoded + ordinal_features_encoded
y_values = []
x_values = []

for index in np.argsort(model.feature_importances_)[::-1][:20][::-1]:
    x_values.append(features[index])
    y_values.append(model.feature_importances_[index])
fig, ax = plt.subplots()
y_pos = range(len(y_values))
ax.barh(y_pos, y_values)
ax.set_yticks(y_pos)
ax.set_yticklabels(x_values)
plt.show()
test_set = pd.read_csv('test.csv')
test_set.head()
test_set = encode_dataframe(test_set, train=False, 
                           train_set_categorical_encoded_means=train_val_categorical_global_means)
# test_set.sample(10)
test_set_selected_features = test_set[categorical_features_encoded + ordinal_features_encoded + numerical_features]
# Problem: some categorical feature values appear in the test set but not in the training or validation set, so they
# do not have a target mean that can be mapped onto the feature values in the test set

# step 1: find out which feature values appar in the test set but not in the training/validation set

# create two emtpy lists
unequal_features = []
unequal_values = []

for feature in categorical_features:
    # create two lists with every categorical feature value in the train and test set, respectively
    # .index return index (in this case name of categorical feature of Series), .tolist() turns it into a list
    train_set_feature_values = train_set[feature].value_counts().index.tolist()
    test_set_feature_values = test_set[feature].value_counts().index.tolist()
    # append the feature and the feature value to a list if it appears in the test set ut not in the training/validation set
    for i in test_set_feature_values:
        if not i in train_set_feature_values:
            unequal_features.append(feature)
            unequal_values.append(i)
            
print(unequal_features)
print(unequal_values)
# step 2: replace the categorical feature values which only appear in the test set with the global SalePrice mean

# create a list of encoded features 
unequal_features_encoded = [feature + '_encoded' for feature in unequal_features]

for i in range(len(unequal_features)):
    # create a new dictionary
    test_categorical_encoding = {}
    # create a new dict (with the encoded feature name as a key) inside the new dict
    test_categorical_encoding[unequal_features_encoded[i]] = {}
    # set the global SalePrice mean as the value for each key
    test_categorical_encoding[unequal_features_encoded[i]][unequal_values[i]] = train_set.SalePrice.mean()
    # replace the feature value in the test set with the dict (which includes the SalePrice mean for the feature value
    # that only appears in the test set)
    test_set_selected_features = test_set_selected_features.replace(test_categorical_encoding)
test_set_selected_features = test_set_selected_features.apply(pd.to_numeric)
test_set_selected_features.sample(5)
submission_predicted = model.predict(test_set_selected_features)
test_set['SalePrice'] = submission_predicted
test_set['SalePrice'] = np.expm1(test_set['SalePrice'])
test_set.head()
submission = test_set[['Id', 'SalePrice']]
submission.head()
submission.to_csv (r'/Users/Helene/desktop/house_prices_sub.csv', index = False, header=True)