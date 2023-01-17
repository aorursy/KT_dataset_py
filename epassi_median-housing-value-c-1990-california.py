import os
import tarfile
import urllib
import pandas as pd

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Create dataset directory, download housing.tgz, extract housing.csv.
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# DataFrame.
def housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
fetch_housing_data()
housing = housing_data()
housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
train_set.head()
test_set.head()
from sklearn.model_selection import train_test_split # btw: much better name for a function that returns something

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
train_set.head()
test_set.head()
housing['median_income'].hist(bins=50)
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
                               labels=[1, 2, 3, 4, 5])

housing['income_cat'].hist()
from sklearn.model_selection import StratifiedShuffleSplit

# Split the data once (into 2 parts), 80/20%, and set the random seed at 42.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude')
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
housing.plot(
    kind='scatter',
    x='longitude',
    y='latitude',
    alpha=0.4,
    s=housing['population']/100, # 100 is arbitrary. Adjust to get the desired scale.
    label='population',
    figsize=(11,10), # Adjust till we get the shape of California.
    c='median_house_value', # Set color based on median_house_value.
    cmap=plt.get_cmap('jet'), # Use the "jet" color map.
    colorbar=True
    )
housing.corr()['median_house_value'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(
    kind='scatter',
    x='median_income',
    y='median_house_value',
    alpha=0.1
    )
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households'] 
housing.corr()['median_house_value'].sort_values(ascending=False)
housing.plot(
    kind='scatter',
    x='bedrooms_per_room',
    y='median_house_value',
    alpha=0.1
    )
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
# 1. Get rid of the corresponding individuals.
# housing.dropna(subset['total_bedrooms'])

# 2. Get rid of the whole attribute/column.
# housing.drop('total_bedrooms', axis=1)

# 3. Fill in the missing features with a sensible value (such as the median).
# median = housing['total_bedrooms'].median()
# housing['total_bedrooms'].fillna(median, inplace=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median') # Same as using Pandas fillna() with the median
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
housing_tr
housing_cat = housing[['ocean_proximity']]
housing_cat.head(10)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
ordinal_encoder.categories_
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder.categories_
from sklearn.base import BaseEstimator, TransformerMixin

# Indices.
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or *kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Create the pipeline for numeric transformations on housing_num. 
num_pipeline = Pipeline([
    # This is a list of tuples.
    # Each tuple is a name/estimator pair.
    # The name is our own unique ID that will come in handy for hyperparameter tuning.
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

# Create the full pipeline for both numerical and categorical data.
num_attribs = list(housing_num) # returns housing_num column names
cat_attribs = ['ocean_proximity']
full_pipeline = ColumnTransformer([
    # More name/estimator tuples.
    ('num', num_pipeline, num_attribs), # num_pipeline is just another transformation
    ('cat', OneHotEncoder(), cat_attribs)
])

# Save the data we'll use for training our model.
housing_prepared = full_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
lin_reg.predict(some_data_prepared)
list(some_labels)
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
housing_predictions
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard deviation:', scores.std())
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)    
display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
housing_predictions
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV

# Array of hyperparameter combinations.
# They keys are hyperparameters; the arrays are the values we want to test.
# 18 total configurations.
param_grid = [
    # 3 × 4 = 12 value combinations
    {
        'n_estimators': [3, 10, 30],
        'max_features': [2, 4, 6, 8]
    },
    # 1 × 2 × 3 = 6 value combinations
    {
        'bootstrap': [False],
        'n_estimators': [3, 10],
        'max_features': [2, 3, 4]
    }
]

forest_reg = RandomForestRegressor()

# 5-fold cross validator.
# Will check all 18 configs 5 times.
# This might take a while.
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

# Get the highest performing hyperparameter values.
grid_search.best_params_
grid_search.best_estimator_
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
# Grab our best model.
final_model = grid_search.best_estimator_

# Get the test set and remove the value we want to predict (median_house_value).
X_test = strat_test_set.drop('median_house_value', axis=1)

# Keep a copy of the labels so we can measure performance.
y_test = strat_test_set['median_house_value'].copy()

# Clean up the data with our pipeline.
X_test_prepared = full_pipeline.transform(X_test)

# Make the predictions.
final_predictions = final_model.predict(X_test_prepared)

# Measure performance.
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, 
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
from sklearn.externals import joblib

joblib.dump(final_model, 'median-house-value.joblib')
print(full_pipeline.named_transformers_['cat'].get_feature_names())
housing.describe()
training_stats = housing.describe().copy().iloc[1:3, :]
training_stats
rooms_per_household = housing.iloc[:, rooms_ix] / housing.iloc[:, households_ix]
population_per_household = housing.iloc[:, population_ix] / housing.iloc[:, households_ix]
bedrooms_per_room = housing.iloc[:, bedrooms_ix] / housing.iloc[:, rooms_ix]

training_stats['rooms_per_household'] = [rooms_per_household.mean(), rooms_per_household.std()]
training_stats['population_per_household'] = [population_per_household.mean(), population_per_household.std()]
training_stats['bedrooms_per_room'] = [bedrooms_per_room.mean(), bedrooms_per_room.std()]
training_stats
qa = X_test.head(10).copy()
qa['PREDICTION'] = final_predictions[:10]
qa