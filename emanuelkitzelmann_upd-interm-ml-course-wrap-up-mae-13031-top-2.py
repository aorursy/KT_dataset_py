import pandas as pd
#from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Read the data
train_data = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

# Update 1: chose all categorical columns instead of only low cardinality features (see above)
categorical_cols = [cname for cname in train_data.columns if train_data[cname].dtype == "object"]
# low_cardinality_cols = [cname for cname in train_data.columns if train_data[cname].nunique() < 10 
#                        and train_data[cname].dtype == "object"]

# Keep selected cols
my_cols = categorical_cols + numeric_cols
X = train_data[my_cols].copy()
X_test = test_data[my_cols].copy()

# We use cross validation and hence do not need to split into train and validation set
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
#                                                               random_state=0)
# Preprocessing for numerical data
# As mentioned above, we don't impute missing numerical data, 
# so this is out-commented
#numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),   # !!! important to keep numeric data
        ('cat', categorical_transformer, categorical_cols)
    ])

# instantiating the XGBoost model and the pipeline
model = XGBRegressor(learning_rate=0.01, random_state=0)
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
# The initial parameter search space. 
# We coarsely search for the best number of estimators within 200 to 2000 in steps of 200.
parameter_space = {
    'model__n_estimators': [n for n in range(200, 2001, 200)]
}
print("Initial parameter search space: ", parameter_space)

# Initializing the grid search.
folds = KFold(n_splits=5, shuffle=True, random_state=0)
grid_search = GridSearchCV(my_pipeline, param_grid=parameter_space, 
                           scoring='neg_mean_absolute_error', cv=folds)

# First search round.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Fix n_estimators to the best found value
parameter_space['model__n_estimators'] = [grid_search.best_params_['model__n_estimators']]

# We add max_depth and min_child_weight with possible values 1, 4, 7 each to the search.
parameter_space['model__max_depth'] = [x for x in [1, 4, 7]]
parameter_space['model__min_child_weight'] = [x for x in [1, 4, 7]]
print("Updated parameter space: ", parameter_space)

# Search.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Fine tuning.
fine_tune_range = [-1, 0, 1]
# Update the parameter space for fine tuning.
parameter_space['model__max_depth'] = [grid_search.best_params_['model__max_depth'] + i 
                                       for i in fine_tune_range]
parameter_space['model__min_child_weight'] = [grid_search.best_params_['model__min_child_weight'] + i 
                                              for i in fine_tune_range]
print("Updated parameter space: ", parameter_space)

# Search.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# We now have fixed the final values for max_depth and min_child_weight 
# and update the search space accordingly.
parameter_space['model__max_depth'] = [grid_search.best_params_['model__max_depth']]
parameter_space['model__min_child_weight'] = [grid_search.best_params_['model__min_child_weight']]

# Add subsample and colsample_bytree with possible values 0.6 and 0.9 each.
parameter_space['model__subsample'] = [x/10 for x in [6, 9]]
parameter_space['model__colsample_bytree'] = [x/10 for x in [6, 9]]
print("Updated parameter space: ", parameter_space)

# Search.
grid_search.fit(X, y)
print("Best found parameter values: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Fine tuning for subsample and colsample_bytree and fixing their final values
parameter_space['model__subsample'] = [grid_search.best_params_['model__subsample'] + i/10 
                                       for i in fine_tune_range]
parameter_space['model__colsample_bytree'] = [grid_search.best_params_['model__colsample_bytree'] + i/10 
                                              for i in fine_tune_range]
print("Updated parameter space: ", parameter_space)

grid_search.fit(X, y)

parameter_space['model__subsample'] = [grid_search.best_params_['model__subsample']]
parameter_space['model__colsample_bytree'] = [grid_search.best_params_['model__colsample_bytree']]

# Parameter values so far...
print("Fixed parameter values: ", parameter_space)

import xgboost as xgb
X_enc = pd.get_dummies(X)
dtrain = xgb.DMatrix(X_enc, label=y)

# Setting up parameter dict with found optimal values
params = {
    'max_depth': parameter_space['model__max_depth'][0],
    'min_child_weight': parameter_space['model__min_child_weight'][0],
    'eta': 0.01, # eta is the learning rate
    'subsample': parameter_space['model__subsample'][0],
    'colsample_bytree': parameter_space['model__colsample_bytree'][0]
}
print(params)
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=10000,
    seed=0,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=100
)
#print(cv_results)

mae = cv_results['test-mae-mean'].min()
opt_n_estimators = cv_results['test-mae-mean'].argmin()

print("Optimal number of estimators: ", opt_n_estimators)
print("Score: ", mae)

params = {
    'n_estimators': opt_n_estimators,
    'learning_rate': 0.01, 
    'max_depth': parameter_space['model__max_depth'][0],
    'min_child_weight': parameter_space['model__min_child_weight'][0],
    'subsample': parameter_space['model__subsample'][0],
    'colsample_bytree': parameter_space['model__colsample_bytree'][0]
}

final_model = XGBRegressor(**params, random_state=0)
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', final_model)])

#print(final_pipeline.named_steps['model'])

final_pipeline.fit(X, y)
# Compute predictions on test data and save to file.
preds_test = final_pipeline.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
