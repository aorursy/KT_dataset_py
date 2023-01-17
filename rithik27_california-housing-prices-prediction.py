# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

#Common Imports
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#To plot pretty figures
%matplotlib inline
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#Ignore Warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

np.random.seed(42)
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    return

fetch_housing_data()
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing_data = load_housing_data()
print(housing_data.head())
#Getting the info about the dataset
print(housing_data.info())
numerical_attributes = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
                        "households", "median_income"]
categorical_attributes = ["ocean_proximity"]
print(housing_data.describe())
#Visualising The Data
%matplotlib inline
housing_data.hist(bins=50, figsize=(20,15))
plt.show()
from sklearn.model_selection import StratifiedShuffleSplit

housing_data["income_cat"] = np.ceil(housing_data["median_income"]/1.5)
housing_data["income_cat"].where(housing_data["income_cat"]>5, 5.0, inplace=True)
data = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in data.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

# Dropping the "income_cat" attribute
for x in (strat_train_set, strat_test_set):
    x.drop("income_cat", axis=1, inplace=True)

training_set = strat_train_set.copy()
testing_set = strat_test_set.copy()
training_set.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, 
                  s=training_set["population"]/100, label="Population", c="median_house_value",
                  cmap=plt.get_cmap("jet"), colorbar=True)
correlation_matrix = training_set.corr()
print(correlation_matrix["median_house_value"].sort_values(ascending=True))
from pandas.plotting import scatter_matrix
scatter_matrix(training_set[["median_house_value", "total_rooms", "housing_median_age", "median_income"]], figsize=(12, 8))
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

predictors = training_set.drop("median_house_value", axis=1)
labels = training_set["median_house_value"].copy()
imputer = SimpleImputer(strategy="median")
rooms_ix, bedrooms_ix, population_ix, household_ix = [
    list(predictors.columns).index(col)
    for col in ("total_rooms", "total_bedrooms", "population", "households")]
class AttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribute_adder', AttributesAdder()),
            ('Scaler', StandardScaler()),
            ])

categorical_pipeline = Pipeline([
            ('encoding', OneHotEncoder()),
            ])

full_pipeline = ColumnTransformer([
                ("num", numerical_pipeline, numerical_attributes),
                ("cat", categorical_pipeline, categorical_attributes),
            ])
prepared_data = full_pipeline.fit_transform(predictors)
print(prepared_data)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(prepared_data, labels)
predictions = lin_reg.predict(prepared_data)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(lin_reg, prepared_data, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(prepared_data)
P_reg = LinearRegression()
P_reg.fit(X_poly, labels)
predictions = P_reg.predict(X_poly)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(P_reg, X_poly, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(prepared_data, labels)
predictions = ridge_reg.predict(prepared_data)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(ridge_reg, prepared_data, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1, max_iter=100000, tol=0.01)
lasso_reg.fit(prepared_data, labels)
predictions = lasso_reg.predict(prepared_data)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(lasso_reg, prepared_data, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.svm import LinearSVR
svm_reg = LinearSVR(epsilon=10, C=10000)
svm_reg.fit(prepared_data, labels)
predictions = svm_reg.predict(prepared_data)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(svm_reg, prepared_data, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.svm import SVR
svm_reg = SVR(kernel="poly", degree=2, C=10000, epsilon=0.1)
svm_reg.fit(prepared_data, labels)
predictions = svm_reg.predict(prepared_data)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(svm_reg, prepared_data, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(prepared_data, labels)
predictions = tree_reg.predict(prepared_data)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(svm_reg, prepared_data, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(prepared_data, labels)
predictions = forest_reg.predict(prepared_data)
rmse = np.sqrt(mean_squared_error(labels, predictions))
validation_scores = cross_val_score(svm_reg, prepared_data, labels, scoring="neg_mean_squared_error", cv=10)
validation_scores = np.sqrt(-validation_scores)  #root mean squared error
print("ROOT MEAN SQUARED ERROR = ", rmse)
print("VALIDATION SCORES = ", validation_scores)
from sklearn.model_selection import GridSearchCV

param_grid = [
            {'n_estimators': [3, 10, 20, 25, 30], 'max_features': [2, 4, 6, 8], 'bootstrap': [True, False]}
            ]
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
grid_search.fit(prepared_data, labels)
print("The best parameters for Random Forest Regressor is : ", grid_search.best_params_)
scores = grid_search.cv_results_
for mean_score, params in zip(scores["mean_test_score"], scores["params"]):
    print(np.sqrt(-mean_score), params)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = [
            {'n_estimators': randint(low=1, high=200),
             'max_features': randint(low=1, high=8),
             'bootstrap': [True, False]}
            ]
random_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, cv=5,
                                   n_iter=10, scoring="neg_mean_squared_error", random_state=42)
random_search.fit(prepared_data, labels)
print("The best parameters for Random Forest Regressor is : ", random_search.best_params_)
scores = random_search.cv_results_
for mean_score, params in zip(scores["mean_test_score"], scores["params"]):
    print(np.sqrt(-mean_score), params)
final_model = random_search.best_estimator_
x_test = testing_set.drop("median_house_value", axis=1)
y_test = testing_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("The root mean squared error of the test data is :", final_rmse)