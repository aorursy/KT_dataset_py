import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
housing = pd.read_csv("../input/california-housing-prices/housing.csv")

housing.head(3)
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
housing.hist(bins=50, figsize=(20, 15))

plt.show()
# median income looks like an imp feature



housing['median_income'].hist()
# dividing the income category to limit the number income category

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)

# putting everything above 5th category as 5th category

housing['income_cat'].where(housing['income_cat'] < 5, other=5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=29)



for train_index, test_index in split.split(housing, housing['income_cat']):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts() / len(housing)
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
# experimenting: with random sampling now



from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing,  test_size=0.2, random_state=29)
def income_cat_proportions(data):

    return data['income_cat'].value_counts() / len(data)





comparing_props = pd.DataFrame({

    "Overall Props": income_cat_proportions(housing),

    "Random": income_cat_proportions(test_set),

    "Stratified": income_cat_proportions(strat_test_set)

}).sort_index()



comparing_props["random %error"] = 100 * comparing_props["Random"] / comparing_props["Overall Props"] - 100

comparing_props["strat. %error"] = 100 * comparing_props["Stratified"] / comparing_props["Overall Props"] - 100

comparing_props
for items in (strat_train_set, strat_test_set):

    items.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

            s=housing['population']/100, label="population", figsize=(12,8),

            c="median_house_value", cmap=plt.get_cmap("jet"), sharex=False)



plt.legend()
# pandas has corr method for calculating correlations

corr_matrix = housing.corr()



corr_matrix["median_house_value"].sort_values(ascending=False)
# other approach it to use the scatter plot in a A vs B fashion

# problem with this is that (for N features, there will be N^2 plots)



imp_attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]



from pandas.plotting import scatter_matrix



scatter_matrix(housing[imp_attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

plt.axis([0, 16, 0, 550000])
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"] = housing["population"]/housing["households"]

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)

plt.axis([0, 5, 0, 520000])

plt.show()
housing.describe()
housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()
# when calculating imputng value on your own

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()



median = housing["total_bedrooms"].median()

sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)

sample_incomplete_rows
# when using Scikit-Learn's Imputer class

from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)



imputer.fit(housing_num)
# Imputer basically computes across all the attributes, so if you wanna see this across all the attributes, just call statistics_ method

imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# cross check for missing value

housing_tr[housing_tr.isnull().any(axis=1)]
housing_tr.head()
housing_cat = housing["ocean_proximity"]

housing_cat.head(10)
# using pandas's own factorize() method to convert them into categorical features

housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]
housing_categories
# using Scikit-Learn's OneHotEncoder



from sklearn.preprocessing import OneHotEncoder



encoder = OneHotEncoder()

housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(1, -1))
housing_cat_1hot
# since 1 hot encoder returns a sparse matrix, need to change it to a dense array

housing_cat_1hot.toarray()
from sklearn.base import BaseEstimator, TransformerMixin



#column indexes

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    

    def __init__(self, add_bedrooms_per_room = True):

        self.add_bedrooms_per_room = add_bedrooms_per_room

        

    def fit(self, X, y=None):

        return self # nothing to do here

    

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing.columns)+["rooms_per_household", 

                                                                                           "population_per_household"])

housing_extra_attribs.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ("imputer", SimpleImputer(strategy="median")),

    ("attribs_adder", CombinedAttributesAdder()),

    ("std_scaler", StandardScaler())

])



housing_num_tr = num_pipeline.fit_transform(housing_num)

housing_num_tr
class DataFrameSelector(BaseEstimator, TransformerMixin):

    

    def __init__(self, attribute_names):

        self.attibute_names = attribute_names

        

    def fit(self, X, y=None):

        return self # do nothing

    

    def transform(self, X, y=None):

        return X[self.attibute_names].values
# complete Pipeline



num_attribs = list(housing_num.columns)

cat_attribs = ["ocean_proximity"]



num_pipeline = Pipeline([

    ("selector", DataFrameSelector(num_attribs)),

    ("imputer", SimpleImputer(strategy="median")),

    ("attribs_adder", CombinedAttributesAdder()),

    ("std_scaler", StandardScaler())

])



cat_pipeline =Pipeline([

    ("selector", DataFrameSelector(cat_attribs)),

    ("cat_encoder", OneHotEncoder(sparse=False))

])
from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

    ('num_pipeline', num_pipeline),

    ('cat_pipeline', cat_pipeline)

])
housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
# trying the full pipeline on a few training instances



some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]



some_data_prepared = full_pipeline.transform(some_data)
print("Prediction: ", lin_reg.predict(some_data_prepared))

print("Actual Labels: ", list(some_labels))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)



lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)



tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepared, housing_labels, cv=10, scoring="neg_mean_squared_error")



tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("scores: ", scores)

    print("mean: ", scores.mean())

    print("std deviation: ", scores.std())

    

    

display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, cv=10, scoring="neg_mean_squared_error")



lin_rmse_scores = np.sqrt(-lin_scores)



display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(random_state=29)

forest_reg.fit(housing_prepared, housing_labels)
housing_pred = forest_reg.predict(housing_prepared)



forest_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, cv=10, scoring="neg_mean_squared_error")



forest_rmse_scores = np.sqrt(-forest_scores)



display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}

]



rf_reg = RandomForestRegressor()



grid_search = GridSearchCV(rf_reg, param_grid, cv=5, scoring="neg_mean_squared_error")



grid_search.fit(housing_prepared, housing_labels)
# to get the best combination of hyperparameters

grid_search.best_params_
# to get the best estimators directly

grid_search.best_estimator_
cv_res = grid_search.cv_results_



for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):

    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



params_distibs = {

    'n_estimators': randint(low=1, high=200),

    'max_features': randint(low=1, high=8),

}



rf_reg = RandomForestRegressor(random_state=29)



rnd_search = RandomizedSearchCV(rf_reg, param_distributions=params_distibs, n_iter=10, 

                                cv=5, scoring="neg_mean_squared_error", random_state=29)



rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_



for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = cat_pipeline.named_steps["cat_encoder"]

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