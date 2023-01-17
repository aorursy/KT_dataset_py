import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')
!wget https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz
#Unzipping
!tar --gunzip --extract --verbose --file=housing.tgz
!ls -lh *csv
housing = pd.read_csv('housing.csv')
housing.head(15)
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
%matplotlib inline
housing.hist(bins = 100, figsize = (10, 15))
plt.show()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.20, random_state=42)
print(len(train_set), len(test_set))
print(16512+4128)
train_set.head(2)
test_set.head(2)
housing["income_category"] = pd.cut(housing["median_income"], 
                                    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                    labels = [1, 2, 3, 4, 5])
print(housing["income_category"].value_counts())
housing.income_category.hist()
housing.head()
test_set.index[1]
train_set.index[1]
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(housing, housing["income_category"]):
  print("Train: ", train_index, "Test: ", test_index)
  strat_train_set = housing.loc[train_index]
  strat_test_set = housing.loc[test_index]

def income_cat_proportion(data):
  return data["income_category"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportion(housing),
    "Stratified": income_cat_proportion(strat_train_set),
    "Random": income_cat_proportion(test_set)
}).sort_index()

compare_props["Rand. %error"] = 100*compare_props["Random"] / compare_props["Overall"]
compare_props["Strat. %error"] = 100*compare_props["Stratified"] / compare_props["Overall"]
train_set.head(2)
test_set.head(2)
housing_.head()
compare_props
strat_train_set
strat_test_set
16512+4128
for sett in (strat_train_set, strat_test_set):
  sett.drop(["income_category"], axis=1, inplace=True)
strat_train_set
strat_test_set
housing = strat_train_set
housing
housing.plot(x="longitude", y="latitude", kind="scatter", grid= True)
plt.show()
xx = housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
xx
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing_["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
corr_matrix = housing.corr()
corr_matrix
corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot( kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0, 16, 0, 550000])
from google.colab import files
xx = plt.figure()
plt.scatter(x = housing["median_income"], y=housing["median_house_value"], alpha=0.1)
plt.show()

xx.savefig('ok.png')
files.download('ok.png')
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]
corr = housing_.corr()
corr["median_house_value"].sort_values(ascending = False)
housing
strat_train_set.head(2)
strat_train_set = strat_train_set.drop("rooms_per_household", axis = 1)
strat_train_set = strat_train_set.drop("bedrooms_per_room", axis = 1)
strat_train_set = strat_train_set.drop("population_per_household", axis = 1)
strat_train_set.head(2)
strat_test_set.head()
housing_train_data = strat_train_set.drop("median_house_value", axis = 1)
housing_train_labels = strat_train_set["median_house_value"].copy()
housing_train_data
housing_train_labels.head()
strat_test_set
incomplete_rows = housing_train_data[housing_train_data.isnull().any(axis=1)]
incomplete_rows
#incomplete_rows = incomplete_rows.drop("total_bedrooms", axis=1)
#incomplete_rows
meddian = housing_train_data["total_bedrooms"].median()
incomplete_rows["total_bedrooms"].fillna(meddian, inplace=True)
incomplete_rows
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing_train_data.select_dtypes(include=[np.number])
housing_num
null_ros = housing_num[housing_num.isnull().any(axis=1)]
null_ros
imputer.fit(housing_num)
imputer.statistics_
housing_num
X = imputer.transform(housing_num)
housing_train_data_filled = pd.DataFrame(X, index=housing_train_data.index, columns=housing_num.columns)
housing_train_data_filled.loc[incomplete_rows.index.values]
housing_train_data_filled
housing_train_data_filled.index
housing_train_data_filled.head()
#housing_train_data_filled = housing_train_data_filled.sort_index()
#housing_train_data_filled
housing_train_labels
housing[['ocean_proximity']]
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housinng_cat = housing[["ocean_proximity"]]
housing_cat_encoded = ordinal_encoder.fit_transform(housinng_cat)
housing_cat_encoded[0:10]
ordinal_encoder.categories_
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
housing_cat_one_hot_encoded = one_hot_encoder.fit_transform(housinng_cat)
housing_cat_one_hot_encoded
housing_cat_one_hot_encoded.toarray()

one_hot_encoder.categories_
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing_train_data_filled.values)
housing_extra_attribs
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing_train_data_filled.columns)+["rooms_per_household", "population_per_household"],
    index=housing_train_data_filled.index)

housing_extra_attribs
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([('imputer',
                          SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),])
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)
len(housing_num_tr)
housing_train_data.head(2)
from sklearn.compose import ColumnTransformer
# list of numerical column names
num_attribs = list(housing_num)
# list of categorical column names
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing_train_data)
housing_prepared
housing_prepared.shape
len(housing_train_labels)
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(housing_prepared, housing_train_labels)

some_data = housing_train_data.iloc[: 5]
some_data
some_labels = housing_train_labels[0:5]
some_labels
some_data_prepared = full_pipeline.transform(some_data)
predictions = linear_regression.predict(some_data_prepared)
predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error

housing_predictions = linear_regression.predict(housing_prepared)
lin_mse = mean_squared_error(housing_predictions, housing_train_labels)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
lin_mae = mean_absolute_error(housing_predictions, housing_train_labels)
lin_mae
from sklearn.tree import DecisionTreeRegressor
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
decision_tree_regressor.fit(housing_prepared, housing_train_labels)
housing_predictions = decision_tree_regressor.predict(housing_prepared)
tree_mse = mean_squared_error(housing_predictions, housing_train_labels)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(decision_tree_regressor, housing_prepared, housing_train_labels,\
                        scoring = "neg_mean_squared_error", cv = 10)

tree_rmse_scores = np.sqrt(-scores)

tree_rmse_scores
print("Mean of scores: ", tree_rmse_scores.mean())
print("Standard Deviation of Scores: ", tree_rmse_scores.std())
lin_scores = cross_val_score(linear_regression, housing_prepared, housing_train_labels,\
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
lin_rmse_scores
print("Mean of scores: ", lin_rmse_scores.mean())
print("Standard Deviation of Scores: ", lin_rmse_scores.std())
from sklearn.ensemble import RandomForestRegressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(housing_prepared, housing_train_labels)
housing_predictions = random_forest_regressor.predict(housing_prepared)
random_forest_mse = mean_squared_error(housing_train_labels, housing_predictions)
random_forest_rmse = np.sqrt(random_forest_mse)
random_forest_rmse
scores = cross_val_score(linear_regression, housing_prepared, housing_train_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()
from sklearn.model_selection import GridSearchCV

parameters_grid = [
                   {
                       "n_estimators" : [3, 10, 10],
                       "max_features" : [2, 4, 6, 8],
                   },

                   {
                       "bootstrap" : [False],
                       "n_estimators" : [3, 10],
                       "max_features" : [2, 3, 4]
                   }
]


grid_search = GridSearchCV(random_forest_regressor, parameters_grid, cv=5,
                           scoring = "neg_mean_squared_error", return_train_score = True)
grid_search.fit(housing_prepared, housing_train_labels)

grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

params_distribs = {
    "n_estimators" : randint(low = 1, high = 100),
    "max_features" : randint(low = 1, high = 8)
}

rand_search = RandomizedSearchCV(random_forest_regressor, param_distributions = params_distribs,
                                 n_iter=10, cv=5, scoring="neg_mean_squared_error", random_state=42)

rand_search.fit(housing_prepared, housing_train_labels)
cvres = rand_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
strat_test_set
X_test = strat_test_set.drop(["median_house_value"], axis=1)
X_test
grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test
y_test
final_model = grid_search.best_estimator_

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test.values, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
y_test
y_test.index
y_test.values
y_test.values[1]
final_predictions
good = bad = 0

eps = 0.15

for i in range(len(y_test)):

  if ((abs(y_test.values[i] - final_predictions[i]))/y_test.values[i]) <= eps :
    good += 1
  else:
    bad += 1

print("Good Prediction # : ", good)
print("Bad Prediction # : ", bad)