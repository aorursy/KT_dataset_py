# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Where to save the figures

PROJECT_ROOT_DIR = "."

CHAPTER_ID = "end_to_end_project"

IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

os.makedirs(IMAGES_PATH, exist_ok=True)



def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
import os

import tarfile

from six.moves import urllib



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
import pandas as pd



def load_housing_data(housing_path=HOUSING_PATH):

    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)
housing = load_housing_data()

housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))

save_fig("attribute_histogram_plots")

plt.show()
# to make this notebook's output identical at every run

np.random.seed(42)
import numpy as np



# For illustration only. Sklearn has train_test_split()

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)

len(train_set)
len(test_set)
from zlib import crc32



def test_set_check(identifier, test_ratio):

    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32



def split_train_test_by_id(data, test_ratio, id_column):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))

    return data.loc[~in_test_set], data.loc[in_test_set]
import hashlib



def test_set_check(identifier, test_ratio, hash=hashlib.md5):

    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
def test_set_check(identifier, test_ratio, hash=hashlib.md5):

    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio
housing_with_id = housing.reset_index()   # adds an `index` column

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
test_set.head()
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()
housing["median_income"].hist()
housing["income_cat"] = pd.cut(housing["median_income"],

                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],

                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts()
housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
housing["income_cat"].value_counts() / len(housing)
def income_cat_proportions(data):

    return data["income_cat"].value_counts() / len(data)



train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



compare_props = pd.DataFrame({

    "Overall": income_cat_proportions(housing),

    "Stratified": income_cat_proportions(strat_test_set),

    "Random": income_cat_proportions(test_set),

}).sort_index()

compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100

compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props
for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")

save_fig("bad_visualization_plot")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

save_fig("better_visualization_plot")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

    s=housing["population"]/100, label="population", figsize=(10,7),

    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

plt.legend()

save_fig("housing_prices_scatterplot")
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas

from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))

save_fig("scatter_matrix_plot")
housing.plot(kind="scatter", x="median_income", y="median_house_value",

             alpha=0.1)

plt.axis([0, 16, 0, 550000])

save_fig("income_vs_house_value_scatterplot")
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set

housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

sample_incomplete_rows
median = housing["total_bedrooms"].median()

sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)

# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,

                          index = list(housing.index.values))
housing_tr.loc[sample_incomplete_rows.index.values]
imputer.strategy
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.head()
housing_cat = housing[["ocean_proximity"]]

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



# column index

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

                         bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)

            
housing_extra_attribs = pd.DataFrame(

    housing_extra_attribs,

    columns=list(housing.columns)+["rooms_per_household", "population_per_household"])

housing_extra_attribs.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

    ])



housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr
from sklearn.compose import ColumnTransformer



num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attribs),

    ("cat", OneHotEncoder(), cat_attribs),

])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape
#select and train a model

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
some_data= housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)



print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
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
#fine tune model
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
from sklearn.model_selection import cross_val_score



forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,

                                scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators':[3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2, 3, 4]},

]

forest_reg = RandomForestRegressor()



grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
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
#We can compute a 95% confidence interval for the test RMSE:
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test)**2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc = squared_errors.mean(), scale=stats.sem(squared_errors)))

                                   