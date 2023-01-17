import os

import tarfile

from six.moves import urllib

import sys

print(sys.path)

DOWNLOAD_ROOT = "https://github.com/killakalle/ageron_handson-ml/tree/master/"

HOUSING_PATH = "datasets/housing"

HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    if not os.path.isdir(housing_path):

        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)

    housing_tgz = tarfile.open(tgz_path)

    housing_tgz.extractall(path=housing_path)

    housing_tgz.close()
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

housing.hist(bins=100, figsize=(20,15))

plt.show()
import numpy as np

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]
train_set,test_set = split_train_test(housing,0.2)

print(len(train_set), "train +", len(test_set), "test")
# use hash to make testset



import hashlib

def test_set_check(identifier, test_ratio, hash):

    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio



def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_:test_set_check(id_,test_ratio, hash))

    return data.loc[~in_test_set], data.loc[in_test_set]
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]

#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts()
for set in (strat_train_set, strat_test_set):

    set.drop(["income_cat"], axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100, label="population",

            c="median_house_value",cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
import pandas as pd

attributes = ["median_house_value", "median_income", "total_rooms",

"housing_median_age"]

pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# experimenting with attributes combinations

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
# remove labels from training data

housing = strat_train_set.drop("median_house_value", axis=1)

# prepare a separate array for labels

housing_labels = strat_train_set["median_house_value"].copy()
# some data do not have no of bedrooms field



# option 1: get rid of corresponding districts

#housing.dropna(subset= ["total_bedrooms"])



#option 2: get rid of whole attribute

#housing.drop("total_bedrooms", axis=1)



# option 3: replace missing values by median etc

#median = housing["total_bedrooms"].median()

#housing["total_bedrooms"].fillna(median)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy = "median")

# since median calculated on numerical values only we need to get rid of ocean proximity text

housing_num = housing.drop("ocean_proximity", axis = 1)

imputer.fit(housing_num)

#it calculates median of each attribute and stores it in a statistics_ instance

#it is safe to apply imputer to all attributes in case some other have missing values
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)

#returns a plain NumPy array
# change back to pandas dataframe

import pandas

housing_tr = pandas.DataFrame(X, columns = housing_num.columns)
# converting text labels to numbers

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

housing_cat = housing["ocean_proximity"]

housing_cat_encoded = encoder.fit_transform(housing_cat)

housing_cat_encoded
print(encoder.classes_)
#perform oneHotEncodeing

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))

housing_cat_1hot
#perform both labelEncoding and one hot encoding in one step

from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing_cat)

housing_cat_1hot
# custom transformer for adding extra attributes

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True):

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

        population_per_household = X[:, population_ix] / X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
# we need to use scikit pipelines for all transformations

# scikit does not handle pandas dataframe so convert it to numpy array



# from sklearn.base import BaseEstimator, TransformerMixin



# class DataFrameSelector(BaseEstimator, TransformerMixin):

#     def __init__(self, attribute_names):

#         self.attribute_names = attribute_names

#     def fit(self, X, y=None):

#         return self

#     def transform(self, X)

#         return X[self.attribute_names].values





# writing custom LabelBinarizer to handle three arguments

from sklearn.base import TransformerMixin

class MyLabelBinarizer(TransformerMixin):

    def __init__(self, *args, **kwargs):

        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x,y=0):

        self.encoder.fit(x)

        self.classes_, self.y_type_, self.sparse_input_ = self.encoder.classes_, self.encoder.y_type_, self.encoder.sparse_input_

        return self

    def transform(self,x,y=0):

        return self.encoder.transform(x)
# we need to use scikit pipelines for all transformations

# scikit does not handle pandas dataframe so convert it to numpy array



from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values



# use featureUnion to make single pipe line for two different pipelines that can be run in parallel

from sklearn.pipeline import FeatureUnion

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



num_pipeline= Pipeline([

    ('selector', DataFrameSelector(num_attribs)),

    ('SimpleImputer', SimpleImputer(strategy='median')),

    ('attribs_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler()),

])

cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attribs)),

    ('label_binarizer', MyLabelBinarizer()),

])

full_pipeline = FeatureUnion(transformer_list=[

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

])
housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared

housing_prepared.shape
# actual training of various models



from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]

some_labels = housing_labels[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t", list(some_labels))
# measuring performance

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
# bad performance: lets try a more complex model

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
# smells like overfitting

# use cross validation to verify

from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv = 10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean: ", scores.mean())

    print("Standard Deviation: ", scores.std())

display_scores(tree_rmse_scores)    
#not good: get these scores for linear regression

scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring = 'neg_mean_squared_error', cv = 10)

lin_rmse_scores = np.sqrt(-scores)

display_scores(lin_rmse_scores)

#lets try RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

forest_reg  = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring = "neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)
# lets try Support Vector Machine

from sklearn.svm import SVR

svr = SVR(kernel = 'rbf',C = 1.2, gamma='auto')

svr.fit(housing_prepared, housing_labels)

scores = cross_val_score(svr, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv= 10)

svm_rmse_scores = np.sqrt(-scores)

display_scores(svm_rmse_scores)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = [

{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

]



grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')



grid_search.fit(housing_prepared, housing_labels)



grid_search.best_estimator_

from scipy.stats import expon, reciprocal

# use randomized search cv 

param_distribs =  {

        'kernel': ['linear', 'rbf'],

        'C': reciprocal(20, 200000),

        'gamma': expon(scale=1.0),

    }

svm_reg = SVR()

rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,

                                n_iter=50, cv=5, scoring='neg_mean_squared_error',

                                verbose=2, n_jobs=4, random_state=42)

rnd_search.fit(housing_prepared, housing_labels)

# add transformer to pipeline to select most important features

from sklearn.base import BaseEstimator, TransformerMixin



def indices_of_top_k(arr,k):

    return np.array(np.argpartition(np.array(arr), -k)[-k:])



class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances,k):

        self.feature_importances = feature_importances

        self.k = k

    def fit(self, X, y=None):

        self.feature_indices = indices_of_top_k(self.feature_importances, self.k)

        return self

    def transform(self, X):

        return X[:, self.feature_indices]
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_one_hot_attribs = list(encoder.classes_)

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
# lets say we want 6 most important features

k = 6

top_k_feature_indices = indices_of_top_k(feature_importances, k)

np.array(attributes)[top_k_feature_indices]
# make a new pipeline with top feature selector

prep_and_feature_selection_pipe = Pipeline([

        ('preparation', full_pipeline),

        ('feature_selection', TopFeatureSelector(feature_importances, k))

])
housing_prepared_with_top_k_features = prep_and_feature_selection_pipe.fit_transform(housing)
# making a full pipeline with prep, feature selection and final prediction

prepare_select_predict_pipe = Pipeline([

    ('preparation', full_pipeline),

    ('feature_selection', TopFeatureSelector(feature_importances, k)),

    ('forest_reg', RandomForestRegressor(**grid_search.best_params_))

])

grid_search.best_params_
prepare_select_predict_pipe.fit(housing, housing_labels)
some_data = housing.iloc[:4]

some_labels = housing_labels.iloc[:4]



print("Predictions:\t", prepare_select_predict_pipe.predict(some_data))

print("Labels:\t\t", list(some_labels))
param_grid = [{

    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],

    'feature_selection__k': list(range(1, len(feature_importances) + 1))

}]



grid_search_prep = GridSearchCV(prepare_select_predict_pipe, param_grid, cv=5,

                                scoring='neg_mean_squared_error', verbose=2, n_jobs=1)

grid_search_prep.fit(housing, housing_labels)
# evaluating oon test set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis = 1)

y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)

final_rmse = np.sqrt(final_mse)

print(final_rmse)