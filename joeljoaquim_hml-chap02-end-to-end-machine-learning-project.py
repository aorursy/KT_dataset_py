# To support both python 2 and python 3

from __future__ import division, print_function, unicode_literals



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

LOCAL = False



def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):

    if not LOCAL:

        return

    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)

    print("Saving figure", fig_id)

    if tight_layout:

        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")
os.listdir("../input")
import os

import tarfile

from six.moves import urllib



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"

if LOCAL:

    HOUSING_PATH = os.path.join("datasets", "housing")

else:

    HOUSING_PATH = "../input/california-housing-prices"

HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"



def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    if not LOCAL:

        return

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

# save_fig("attribute_histogram_plots")

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

print(len(train_set), "train +", len(test_set), "test")
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
# def test_set_check(identifier, test_ratio, hash=hashlib.md5):

#     return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio
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
import matplotlib.image as mpimg



if LOCAL:

    california_img=mpimg.imread(PROJECT_ROOT_DIR + '/images/end_to_end_project/california.png')

else:

    california_img=mpimg.imread("../input/california-housing-feature-engineering/california.png")

ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),

                       s=housing['population']/100, label="Population",

                       c="median_house_value", cmap=plt.get_cmap("jet"),

                       colorbar=False, alpha=0.4,

                      )

plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,

           cmap=plt.get_cmap("jet"))

plt.ylabel("Latitude", fontsize=14)

plt.xlabel("Longitude", fontsize=14)



prices = housing["median_house_value"]

tick_values = np.linspace(prices.min(), prices.max(), 11)

cbar = plt.colorbar()

cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)

cbar.set_label('Median House Value', fontsize=16)



plt.legend(fontsize=16)

save_fig("california_housing_prices_plot")

plt.show()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas

from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(18, 12))

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
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",

             alpha=0.2)

plt.axis([0, 5, 0, 520000])

plt.show()
housing.describe()
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set

housing_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

sample_incomplete_rows
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
median = housing["total_bedrooms"].median()

sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3

sample_incomplete_rows
try:

    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+

except ImportError:

    from sklearn.preprocessing import Imputer as SimpleImputer



imputer = SimpleImputer(strategy="median")
housing_num = housing.drop('ocean_proximity', axis=1)

# alternatively: housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,

                          index=housing.index)
housing_tr.loc[sample_incomplete_rows.index.values]
imputer.strategy
housing_tr = pd.DataFrame(X, columns=housing_num.columns,

                          index=housing_num.index)

housing_tr.head()
housing_cat = housing[['ocean_proximity']]

housing_cat.head(10)
try:

    from sklearn.preprocessing import OrdinalEncoder

except ImportError:

    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

housing_cat_encoded[:10]
ordinal_encoder.categories_
try:

    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20

    from sklearn.preprocessing import OneHotEncoder

except ImportError:

    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20



cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
housing_cat_1hot.toarray()
cat_encoder = OneHotEncoder(sparse=False)

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
cat_encoder.categories_
housing.columns
from sklearn.base import BaseEstimator, TransformerMixin



# get the right column indices: safer than hard-coding indices 3, 4, 5, 6

rooms_ix, bedrooms_ix, population_ix, household_ix = [

    list(housing.columns).index(col)

    for col in ("total_rooms", "total_bedrooms", "population", "households")]



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,

                         bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.preprocessing import FunctionTransformer



def add_extra_features(X, add_bedrooms_per_room=True):

    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

    population_per_household = X[:, population_ix] / X[:, household_ix]

    if add_bedrooms_per_room:

        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

        return np.c_[X, rooms_per_household, population_per_household,

                     bedrooms_per_room]

    else:

        return np.c_[X, rooms_per_household, population_per_household]



attr_adder = FunctionTransformer(add_extra_features, validate=False,

                                 kw_args={"add_bedrooms_per_room": False})

housing_extra_attribs = attr_adder.fit_transform(housing.values)
housing_extra_attribs = pd.DataFrame(

    housing_extra_attribs,

    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],

    index=housing.index)

housing_extra_attribs.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),

        ('std_scaler', StandardScaler()),

    ])



housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr
try:

    from sklearn.compose import ColumnTransformer

except ImportError:

    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20
num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared.shape
from sklearn.base import BaseEstimator, TransformerMixin



# Create a class to select numerical or categorical columns 

class OldDataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values
num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



old_num_pipeline = Pipeline([

        ('selector', OldDataFrameSelector(num_attribs)),

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),

        ('std_scaler', StandardScaler()),

    ])



old_cat_pipeline = Pipeline([

        ('selector', OldDataFrameSelector(cat_attribs)),

        ('cat_encoder', OneHotEncoder(sparse=False)),

    ])
from sklearn.pipeline import FeatureUnion



old_full_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", old_num_pipeline),

        ("cat_pipeline", old_cat_pipeline),

    ])
old_housing_prepared = old_full_pipeline.fit_transform(housing)

old_housing_prepared
np.allclose(housing_prepared, old_housing_prepared)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
# let's try the full preprocessing pipeline on a few training instances

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)



print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
some_data_prepared
from sklearn.metrics import mean_squared_error



housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.metrics import mean_absolute_error



lin_mae = mean_absolute_error(housing_labels, housing_predictions)

lin_mae
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
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



forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

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
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

pd.Series(np.sqrt(-scores)).describe()
from sklearn.svm import SVR



svm_reg = SVR(kernel="linear")

svm_reg.fit(housing_prepared, housing_labels)

housing_predictions = svm_reg.predict(housing_prepared)

svm_mse = mean_squared_error(housing_labels, housing_predictions)

svm_rmse = np.sqrt(svm_mse)

svm_rmse
from sklearn.model_selection import GridSearchCV



param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_

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
final_model = grid_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse
from scipy import stats
confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

mean = squared_errors.mean()

m = len(squared_errors)



np.sqrt(stats.t.interval(confidence, m - 1,

                         loc=np.mean(squared_errors),

                         scale=stats.sem(squared_errors)))
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)

tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)

np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)
zscore = stats.norm.ppf((1 + confidence) / 2)

zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)

np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
full_pipeline_with_predictor = Pipeline([

        ("preparation", full_pipeline),

        ("linear", LinearRegression())

    ])



full_pipeline_with_predictor.fit(housing, housing_labels)

full_pipeline_with_predictor.predict(some_data)
my_model = full_pipeline_with_predictor
from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl") # DIFF

#...

my_model_loaded = joblib.load("my_model.pkl") # DIFF
from scipy.stats import geom, expon

geom_distrib=geom(0.5).rvs(10000, random_state=42)

expon_distrib=expon(scale=1).rvs(10000, random_state=42)

plt.hist(geom_distrib, bins=50)

plt.show()

plt.hist(expon_distrib, bins=50)

plt.show()
from sklearn.model_selection import GridSearchCV



param_grid = [

        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},

        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],

         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},

    ]



svm_reg = SVR()

grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)

grid_search.fit(housing_prepared, housing_labels)
negative_mse = grid_search.best_score_

rmse = np.sqrt(-negative_mse)

rmse
grid_search.best_params_
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



# see https://docs.scipy.org/doc/scipy/reference/stats.html

# for `expon()` and `reciprocal()` documentation and more probability distribution functions.



# Note: gamma is ignored when kernel is "linear"

param_distribs = {

        'kernel': ['linear', 'rbf'],

        'C': reciprocal(20, 200000),

        'gamma': expon(scale=1.0),

    }



svm_reg = SVR()

rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,

                                n_iter=50, cv=5, scoring='neg_mean_squared_error',

                                verbose=2, n_jobs=-1, random_state=42)

rnd_search.fit(housing_prepared, housing_labels)
negative_mse = rnd_search.best_score_

rmse = np.sqrt(-negative_mse)

rmse
rnd_search.best_params_
expon_distrib = expon(scale=1.)

samples = expon_distrib.rvs(10000, random_state=42)

plt.figure(figsize=(10, 4))

plt.subplot(121)

plt.title("Exponential distribution (scale=1.0)")

plt.hist(samples, bins=50)

plt.subplot(122)

plt.title("Log of this distribution")

plt.hist(np.log(samples), bins=50)

plt.show()
reciprocal_distrib = reciprocal(20, 200000)

samples = reciprocal_distrib.rvs(10000, random_state=42)

plt.figure(figsize=(10, 4))

plt.subplot(121)

plt.title("Reciprocal distribution (scale=1.0)")

plt.hist(samples, bins=50)

plt.subplot(122)

plt.title("Log of this distribution")

plt.hist(np.log(samples), bins=50)

plt.show()
from sklearn.base import BaseEstimator, TransformerMixin



def indices_of_top_k(arr, k):

    return np.sort(np.argpartition(np.array(arr), -k)[-k:])



class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, k):

        self.feature_importances = feature_importances

        self.k = k

    def fit(self, X, y=None):

        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)

        return self

    def transform(self, X):

        return X[:, self.feature_indices_]
k = 5
top_k_feature_indices = indices_of_top_k(feature_importances, k)

top_k_feature_indices
np.array(attributes)[top_k_feature_indices]
sorted(zip(feature_importances, attributes), reverse=True)[:k]
preparation_and_feature_selection_pipeline = Pipeline([

    ('preparation', full_pipeline),

    ('feature_selection', TopFeatureSelector(feature_importances, k))

])
housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
housing_prepared_top_k_features[0:3]
housing_prepared[0:3, top_k_feature_indices]
prepare_select_and_predict_pipeline = Pipeline([

    ('preparation', full_pipeline),

    ('feature_selection', TopFeatureSelector(feature_importances, k)),

    ('svm_reg', SVR(**rnd_search.best_params_))

])
prepare_select_and_predict_pipeline.fit(housing, housing_labels)
some_data = housing.iloc[:4]

some_labels = housing_labels.iloc[:4]



print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))

print("Labels:\t\t", list(some_labels))
param_grid = [{

    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],

    'feature_selection__k': list(range(1, len(feature_importances) + 1))

}]



grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,

                                scoring='neg_mean_squared_error', verbose=2, n_jobs=4)

grid_search_prep.fit(housing, housing_labels)
grid_search_prep.best_params_