import os

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
housing = pd.read_csv('../input/california-housing-prices/housing.csv')

display(housing.head())
housing.info()
sns.barplot(x=housing["ocean_proximity"].value_counts().index,

            y=housing["ocean_proximity"].value_counts().values)

plt.title("Ocean Proximity")

plt.show()
display(housing.describe())
housing.hist(bins=50, figsize=(20, 15))

plt.show()
sns.distplot(housing['median_income'])

plt.title('Median Income')

plt.show()
housing['income_cat'] = pd.cut(housing["median_income"],

                               bins=[0, 1.5, 3, 4.5, 6, np.inf],

                               labels=[1, 2, 3, 4, 5])

sns.barplot(x=housing["income_cat"].value_counts().index,

         y=housing["income_cat"].value_counts().values)

plt.title("Income Categories")

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
print(housing['income_cat'].value_counts() / len(housing))

print(strat_train_set['income_cat'].value_counts() / len(strat_train_set))

print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

             s=housing["population"]/100, 

             label="population",

             c="median_house_value", cmap=plt.get_cmap("jet"), 

             colorbar=True, figsize=(10,7)) 

plt.legend()

plt.show()
corr_matrix = housing.corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", 

              "total_rooms", "housing_median_age"] 

scatter_matrix(housing[attributes], figsize=(12, 8))

plt.show()
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

plt.show()
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))
housing = strat_train_set.drop("median_house_value", axis=1) 

housing_labels = strat_train_set["median_house_value"].copy()

display(housing.head())
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)

print(imputer.statistics_)

print(housing_num.median().values)
housing_tr = pd.DataFrame(imputer.transform(housing_num), 

                          columns=housing_num.columns)
housing_cat = housing[["ocean_proximity"]]

display(housing_cat.head(10))
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
print(housing_cat_1hot.toarray())
from sklearn.base import BaseEstimator, TransformerMixin 

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6 

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs 

        self.add_bedrooms_per_room = add_bedrooms_per_room 

    def fit(self, X, y=None):

        return self # nothing else to do 

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household] 



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False) 

housing_extra_attribs = attr_adder.transform(housing.values)
display(pd.DataFrame(housing_extra_attribs).head())
from sklearn.pipeline import Pipeline 

from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")), 

                         ('attribs_adder', CombinedAttributesAdder()), 

                         ('std_scaler', StandardScaler()),])

housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num) 

cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), 

                                   ("cat", OneHotEncoder(), cat_attribs),])

housing_prepared = full_pipeline.fit_transform(housing)
pd.DataFrame(housing_prepared).head()
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() 

lin_reg.fit(housing_prepared, housing_labels)
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

print("RMSE:", lin_rmse)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor() 

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

print("RMSE:", tree_rmse)
from sklearn.model_selection import cross_val_score 

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)



def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,

                             scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)

forest_rmse = np.sqrt(forest_mse)

print("RMSE:", forest_rmse)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,

                                scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
# # Save the model

# from sklearn.externals import joblib

# joblib.dump(my_model, "my_model.pkl") 

# # and later...

# my_model_loaded = joblib.load("my_model.pkl")
from sklearn.model_selection import GridSearchCV

param_grid = [

    {'n_estimators': [3, 10, 30], 

     'max_features': [2, 4, 6, 8]}, 

    {'bootstrap': [False], 

     'n_estimators': [3, 10], 

     'max_features': [2, 3, 4]},

]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', 

                           return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)

print(grid_search.best_estimator_)
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

print(feature_importances)
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

print(sorted(zip(feature_importances, attributes), reverse=True))
final_model = grid_search.best_estimator_ 

X_test = strat_test_set.drop("median_house_value", axis=1) 

y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions) 

final_rmse = np.sqrt(final_mse)

print("Final RMSE:", final_rmse)
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

ci = np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, 

                              loc=squared_errors.mean(), 

                              scale=stats.sem(squared_errors)))

print(ci)