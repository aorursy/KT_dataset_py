# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
data = data[data['median_house_value'] < 500001]
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.20, random_state=42, stratify=None)
# Stratify the data: represent all price ranges in both datasets
train_data_features = train_data.drop("median_house_value", axis=1)
train_data_target = train_data['median_house_value'].copy()

test_data_features = test_data.drop("median_house_value", axis=1)
test_data_target = test_data['median_house_value'].copy()
from sklearn.impute import SimpleImputer

train_data_numeric = train_data_features.drop(['ocean_proximity'], axis=1)

imputer = SimpleImputer(strategy="median")
imputer.fit(train_data_numeric)
imputer.statistics_
train_data_numeric.median().values
train_data_clean = imputer.transform(train_data_numeric)
rooms_idx, bedrooms_idx, population_idx, household_idx = (
    list(train_data_features.columns).index(col) for col in ("total_rooms", "total_bedrooms", "population", "households")
)
def add_feature_combinations(X, add_population_per_household=False, add_bedrooms_per_room=False):
    
    rooms_per_household = X[:, rooms_idx] / X[:, household_idx]
    # Add other two features
    
    return np.c_[X, rooms_per_household]
from sklearn.preprocessing import FunctionTransformer

feature_adder = FunctionTransformer(add_feature_combinations, validate=False, kw_args={"add_population_per_household": False, "add_bedrooms_per_room": False})

train_data_features_engineered = feature_adder.fit_transform(train_data_clean)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('feature_adder', FunctionTransformer(add_feature_combinations, validate=False)),
    ('std_scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('1hot', OneHotEncoder())
])
train_data_features.columns
from sklearn.compose import ColumnTransformer

num_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms','total_bedrooms', 'population', 'households', 'median_income']
cat_columns = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ("num", numeric_pipeline, num_columns),
    ("cat", categorical_pipeline, cat_columns)
])

X_train = full_pipeline.fit_transform(train_data_features)
y_train = train_data_target.values
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
X_test = full_pipeline.transform(test_data_features)
y_test = test_data_target.values
# Training error
lin_reg.score(X_train, y_train)
# Generalization error
lin_reg.score(X_test, y_test)
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor()
sgd_reg.fit(X_train, y_train)
sgd_reg.score(X_train, y_train)
sgd_reg.score(X_test, y_test)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
# Training error
tree_reg.score(X_train, y_train)
# Generalization error
tree_reg.score(X_test, y_test)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
rf_reg.score(X_train, y_train)
rf_reg.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}
]
rf_reg = RandomForestRegressor()

grid_search = GridSearchCV(rf_reg, param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
best_rf
# CV error
grid_search.best_score_
# Generalization error
best_rf.score(X_test, y_test)
feature_importances = best_rf.feature_importances_
feature_importances
num_features = num_columns
new_features = ["rooms_per_household"]
oneHot_features = full_pipeline.named_transformers_['cat']['1hot'].categories_[0].tolist()

features = num_features + new_features + oneHot_features
sorted(zip(feature_importances, features), reverse=True)
from joblib import dump

dump(best_rf, 'best_random_forest.joblib')
