import os

import tarfile

from six.moves import urllib

import pandas as pd



DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"

HOUSING_PATH = "datasets/housing"

HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"



def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    if not os.path.isdir(housing_path):

        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)

    #print(tgz_path)

    housing_tgz = tarfile.open(tgz_path, 'r:gz')

    housing_tgz.extractall(path=housing_path)

    housing_tgz.close()



def load_housing_data(housing_path=HOUSING_PATH):

    #csv_path=os.path.join(housing_path, "housing.csv")

    return pd.read_csv('../input/housing.csv')
#fetch_housing_data()

housing = load_housing_data()

housing.head()
housing.info()
housing.ocean_proximity.value_counts()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt

housing.hist(bins=50, figsize=(20,15))

plt.show()
import numpy as np

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set), "train +", len(test_set), "test")
import hashlib

hash = hashlib.md5

hash(np.int64(4)).digest()[-1]
def test_set_check(identifier, test_ratio, hash):

    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio



def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))

    return data.loc[~in_test_set], data.loc[in_test_set]
# we can easily add an identifier column using reset_index

housing_with_id = housing.reset_index()

train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print(len(train_set), "train + ", len(test_set), "test")
# Trim down the number of income categories by scaling and make a greater-than-5 category

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]



# drop the temporarily created income_cat column from both sets

for set_ in (strat_train_set, strat_test_set):

    set_.drop(["income_cat"], axis=1, inplace=True)
# s is the size and c is the color

train_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, 

             label="population", c="median_house_value", figsize=(12,8),

             cmap=plt.get_cmap("jet"), colorbar=True)
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12, 8))
housing = strat_train_set.drop(["median_house_value"], axis=1)

housing_num = housing.drop(["ocean_proximity"], axis=1)

housing_labels = strat_train_set["median_house_value"].copy()
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing["ocean_proximity"])

housing_cat_1hot

# i couldn't figure out how to set this array value as a column to the housing dataframe because it provides too many values.

# Take a look at some rows with missing data

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()

sample_incomplete_rows
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")



imputer.fit(housing_num)



from sklearn.base import BaseEstimator, TransformerMixin



rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6



# Create a class, which inherits from TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self # Nothing else to do

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing_num.values)
housing_extra_attribs
from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin



# Our DataFrameSelector will select the columns from a pandas dataframe

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values

    

# LabelBinarizer changed to support only encoding labels (y)

# not datasets (X) so these days it takes only X not 

# X AND y so this custom class is a hack for that.

class CustomLabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, sparse_output=False):

        self.sparse_output = sparse_output

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        enc = LabelBinarizer(sparse_output=self.sparse_output)

        return enc.fit_transform(X)    



cat_attribs = ["ocean_proximity"]

num_attribs = list(housing.drop(cat_attribs, axis=1))



num_pipeline = Pipeline([

        ('selector', DataFrameSelector(num_attribs)),

        ('imputer', Imputer(strategy="median")),

        ('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

    ])

# we have a separate pipeline for the categorical columns

cat_pipeline = Pipeline([

        ('selector', DataFrameSelector(cat_attribs)),

        ('label_binarizer', CustomLabelBinarizer()),

        ])



full_pipeline = FeatureUnion(transformer_list=[

        ("num_pipeline", num_pipeline),

        ("cat_pipeline", cat_pipeline)

    ])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
# let's try the full pipeline on a few training instances

some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = housing_prepared[:5]

some_data_prepared
print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error



housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)



#scikit0learn cross val expects utilty function

#(Greater is better) rather than cost function (lower better)

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



forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,

                                scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
from sklearn.svm import SVR



svm_reg = SVR()

svm_reg.fit(housing_prepared, housing_labels)



svm_scores = cross_val_score(svm_reg, housing_prepared, housing_labels,

                            scoring="neg_mean_squared_error", cv=10)

svm_rmse_scores = np.sqrt(-svm_scores)

display_scores(svm_rmse_scores)
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

                           scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)
# The best hyperparameter combination found:

grid_search.best_params_
grid_search.best_estimator_
# Let's look at the score of each hyperparameter combination tested during the grid search:

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_)