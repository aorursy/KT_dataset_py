# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
housing = pd.read_csv('../input/housing.csv')

housing.head()

# housing.info()
# Plot the histogram

housing.hist(bins=50, figsize=(20,15))

plt.show()
# Write function to create test data

def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]



# Derive Train set and Test set

train_set, test_set = split_train_test(housing, 0.2)
import hashlib



def test_set_check(identifier, test_ratio, hash):

    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio



def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):

    ids = data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))

    return data.loc[-in_test_set], data.loc[in_test_set]
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)

housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)



from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)

for train_index, test_index in split.split(housing, housing.income_cat):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
housing.income_cat.value_counts() / len(housing)
for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)

    

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 

             s=housing["population"]/100, label="population",figsize=(10,7),

             c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)

plt.legend()
corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
from pandas.tools.plotting import scatter_matrix



attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12,8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
housing["rooms_per_household"] = housing["total_rooms"]/housing['households']

housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']

housing['population_per_household'] = housing['population']/housing['households']
corr_matrix = housing.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
housing = strat_train_set.drop('median_house_value', axis=1)

housing_labels = strat_train_set['median_house_value'].copy()
housing_num = housing.drop("ocean_proximity", axis=1)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

housing_cat = housing['ocean_proximity']

housin_cat_encoded = encoder.fit_transform(housing_cat)

print(encoder.classes_)

housin_cat_encoded
# Using 1 hot encoding

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

housing_cat_1hot = encoder.fit_transform(housin_cat_encoded.reshape(-1,1))

housing_cat_1hot.toarray()
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing_cat)

housing_cat_1hot
from sklearn.base import BaseEstimator, TransformerMixin



rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kwargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    

    def fit(self, X, y=None):

        return self # nothing else to do

    

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]

        population_per_household = X[:, population_ix]/X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]

        

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
# Transformation Pipelines

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, Imputer



num_pipeline = Pipeline([

    ('imputer', Imputer(strategy='median')),

    ('attribs_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler())

])



housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attribute_names].values
num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



from sklearn.base import TransformerMixin #gives fit_transform method for free

class MyLabelBinarizer(TransformerMixin):

    def __init__(self, *args, **kwargs):

        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):

        self.encoder.fit(x)

        return self

    def transform(self, x, y=0):

        return self.encoder.transform(x)



num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_attribs)),

    ('imputer', Imputer(strategy="median")),

    ('attribs_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler())

])



cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attribs)),

    ('label_binarizer', MyLabelBinarizer())

])



from sklearn.pipeline import FeatureUnion



full_pipeline = FeatureUnion(transformer_list=[

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline)

])



housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
# Linear Regression

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels: ", some_labels)



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
# Get scores

from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)



def display_scores(scores):

    print("Scores:", scores)

    print("Mean:",scores.mean())

    print("Standard deviation:", scores.std())

    

display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

forest_scores_rmse = np.sqrt(-forest_scores)

display_scores(forest_scores_rmse)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features':[2,4,6,8]},

    {'bootstrap': [False], 'n_estimators':[3,10], 'max_features': [2, 3, 4]}

]



forest_reg = RandomForestRegressor()



grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error",refit=True)



grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)

print("Best Estimator is")

print(grid_search.best_estimator_)
cvres = grid_search.cv_results_

for mean_scre, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_scre), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_one_hot_attribs = list(encoder.classes_)

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

Y_test = strat_test_set["median_house_value"].copy()



x_test_prepared = full_pipeline.transform(X_test)



final_predictions = final_model.predict(x_test_prepared)



final_mse = mean_squared_error(Y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse
from sklearn.svm import SVR



svm_reg = SVR(kernel="linear",gamma=3.4,C=10)

svm_reg.fit(housing_prepared, housing_labels)
svm_pred = svm_reg.predict(housing_prepared)

svm_mse = mean_squared_error(housing_labels, svm_pred)

svm_rmse = np.sqrt(svm_mse)

svm_rmse
from sklearn.model_selection import RandomizedSearchCV



param_grid = [

    {'kernel': ["rbf","linear"], "gamma": [0.5, 3.0, 20.0, 30.0, 80.0], "C": [10, 100, 1000]}

]



svm_reg = SVR()



random_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring="neg_mean_squared_error",refit=True)



random_search.fit(housing_prepared, housing_labels)



print(random_search.best_params_)

print("Best Estimator is")

print(random_search.best_estimator_)



cvres = random_search.cv_results_

for mean_scre, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_scre), params)
feature_importances = grid_search.best_estimator_.feature_importances_

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_one_hot_attribs = list(encoder.classes_)

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)



final_model = random_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

Y_test = strat_test_set["median_house_value"].copy()



x_test_prepared = full_pipeline.transform(X_test)



final_predictions = final_model.predict(x_test_prepared)



final_mse = mean_squared_error(Y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse
## 