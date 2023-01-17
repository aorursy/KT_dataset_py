import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
Data = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
Data.head()
Data.info()
Data.describe()
Data['ocean_proximity'].value_counts()
Data.hist(bins = 50, figsize = (20,15))
plt.show()
from zlib import crc32
def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

Data_with_id = Data.reset_index() # adds an `index` column
train_set, test_set = split_train_test_by_id(Data_with_id, 0.2, "index")
Data_with_id.info()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(Data, test_size=0.2, random_state=42)
Data['Income Category'] = pd.cut(Data["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
Data['Income Category'].hist()
plt.show()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(Data, Data["Income Category"]):
    strat_train_set = Data.loc[train_index]
    strat_test_set = Data.loc[test_index]
strat_train_set['Income Category'].value_counts()/len(strat_train_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("Income Category", axis=1, inplace=True)
Data = strat_train_set.copy()
Data.plot(kind="scatter", x="longitude", y="latitude")
Data.plot(kind="scatter", x="longitude", y="latitude",alpha = 0.1)
Data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,s=Data["population"]/100, label="population", figsize=(10,7),c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
plt.legend()
corr_matrix = Data.corr()
corr_matrix['median_house_value'].sort_values(ascending = False)
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(Data[attributes], figsize=(12, 8))
plt.show()
Data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
Data['rooms_per_house'] = Data['total_rooms']/Data['households']
Data['bedrooms_per_room'] = Data['total_bedrooms']/Data['total_rooms']
Data['population_per_household'] = Data['population']/Data['households']

corr_matrix = Data.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
Data = strat_train_set.drop("median_house_value", axis=1)
Data_labels = strat_train_set["median_house_value"].copy()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
data_num = Data.drop('ocean_proximity',axis = 1)
imputer.fit(data_num)
''' simple imputer has an attribute statistics_ in which it saves the median values (imputer.statistics_)'''
data_num.median().values
X = imputer.fit(data_num)
Data_tr = pd.DataFrame(X, columns = data_num.columns, index = data_num.index)
data_cat = Data[['ocean_proximity']]
data_cat.head(10)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded_categories = encoder.fit_transform(data_cat)
encoded_categories
encoder.categories_
encoded_categories.toarray()
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        '''we are gateing this parameter since in future we can find out whether this parameter helps the algo'''
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(Data.values)
from sklearn.pipeline import Pipeline
#StandardScaler - feature scaling
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])
housing_num_tr = num_pipeline.fit_transform(data_num)
from sklearn.compose import ColumnTransformer
num_attribs = list(data_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),("cat", OneHotEncoder(), cat_attribs),])
housing_prepared = full_pipeline.fit_transform(Data)
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(housing_prepared,Data_labels)
some_data = Data.iloc[:5]
some_labels = Data_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lr_model.predict(some_data_prepared))
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error
housing_predictions = lr_model.predict(housing_prepared)
lr_mse = mean_squared_error(Data_labels, housing_predictions)
lr_rmse = np.sqrt(lr_mse)
lr_rmse
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
from sklearn.model_selection import cross_val_score
lin_scores = cross_val_score(lr_model, housing_prepared, Data_labels,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
dt_model.fit(housing_prepared, Data_labels)

housing_predictions = dt_model.predict(housing_prepared)
tree_mse = mean_squared_error(Data_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
dt_scores = cross_val_score(dt_model, housing_prepared, Data_labels,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-dt_scores)
display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(housing_prepared, Data_labels)

housing_predictions = rf_model.predict(housing_prepared)
forest_mse = mean_squared_error(Data_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
rf_scores = cross_val_score(rf_model, housing_prepared, Data_labels,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-rf_scores)
display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV
param_grid = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared, Data_labels)
grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse