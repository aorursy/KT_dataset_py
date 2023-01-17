#Fetching file from link



import os

import tarfile

import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"

HOUSING_PATH = os.path.join("datasets", "housing")

HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):

    os.makedirs(housing_path, exist_ok=True)

    tgz_path = os.path.join(housing_path, "housing.tgz")

    urllib.request.urlretrieve(housing_url, tgz_path)

    housing_tgz = tarfile.open(tgz_path)

    housing_tgz.extractall(path=housing_path)

    housing_tgz.close()

    

fetch_housing_data()

print("fetched perfectly")

#Loading data



import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):

    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)



print("Loaded perfectly")
#Cheaking load data

housing = load_housing_data()

housing.head()
#checking info 



housing.info()
#checking catagory of ocean_proximity



housing["ocean_proximity"].value_counts()
#see other feilds and numarical attributes



housing.describe()
#show histogram



import matplotlib.pyplot as plt



housing.hist(bins=50, figsize=(20,15))

plt.show()
#creating tests



import numpy as np



def split_train_test(data, test_ratio):

    shuffled_indices = np.random.permutation(len(data))

    test_set_size = int(len(data) * test_ratio)

    test_indices = shuffled_indices[:test_set_size]

    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]



train_set, test_set = split_train_test(housing, 0.2)

len(train_set)



len(test_set)
train_set, test_set = split_train_test(housing, 0.2)
#splting data



from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)



print("Data splited")

train_set.describe()
import numpy as np

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

housing["income_cat"].hist()
#spliting income perfectly



from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

    

strat_test_set["income_cat"].value_counts() / len(strat_test_set)



#Drop income_cat

for set_ in (strat_train_set, strat_test_set):

    set_.drop("income_cat", axis=1, inplace=True)
#taking copy of training data to play with it

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude")
#make alpha = 0.1 to look much clear

housing.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.1)
#check housing price

housing.plot(kind="scatter",

             x="longitude",

             y="latitude",

             alpha=0.4,

             s=housing["population"]/100,

             label="population", figsize=(10,7),

             c="median_house_value", 

             cmap=plt.get_cmap("jet"), 

             colorbar=True,

)

plt.legend()
#corelations between median house value and other attributes

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
#correlation between different attributes



from pandas.plotting import scatter_matrix



attributes = ["median_house_value",

              "median_income", 

              "total_rooms",

              "housing_median_age"]



scatter_matrix(housing[attributes], figsize=(12, 8))
#zoom in the correlation between median_income and median_house_value

housing.plot(kind="scatter", x="median_income", y="median_house_value",

alpha=0.1)
#some more attributes and corelation with median_house_value



housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]



corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
#copy house median value

housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels = strat_train_set["median_house_value"].copy()
median = housing["total_bedrooms"].median() # option 3

housing["total_bedrooms"].fillna(median, inplace=True)
#filling empty value with median



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
#fit the data in imputer

imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
#getting data after feeling empty data

X = imputer.transform(housing_num)
#converting in pandas datafram from numpy array

housing_tr = pd.DataFrame(X, 

                          columns=housing_num.columns,

                          index=housing_num.index)
#watching text based attribute

housing_cat = housing[["ocean_proximity"]]

housing_cat.head(10)
#converting catagory in numaric values

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

housing_cat_encoded[:10]
#getting orginal catagory 

ordinal_encoder.categories_
#converts catagory on hot and cold catagory

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
#converting in numpy array

housing_cat_1hot.toarray()
#getting orginal catagory

cat_encoder.categories_
#Creating custom transformer



from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    

    def fit(self, X, y=None):

        return self # nothing else to do

    

    def transform(self, X):

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
#Creating transformation piplines



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")),

    ('attribs_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler()),

    ])



housing_num_tr = num_pipeline.fit_transform(housing_num)
#apply transformation pip line

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

    ("num", num_pipeline, num_attribs),

    ("cat", OneHotEncoder(), cat_attribs),])



housing_prepared = full_pipeline.fit_transform(housing)
#training linear regression model

from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)



some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
#training model and testing



some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))

print("Labels:", list(some_labels))
#Checking accuracy

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
#Using decisition tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, housing_labels)



housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
#cross validation. the training set into 10 distinct subsets called folds, then it

#trains and evaluates the Decision Tree model 10 times, picking a different fold for

#evaluation every time and training on the other 9 folds.



from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,

scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
#Creat function for check score



def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())

#Check scor of decision tree

display_scores(tree_rmse_scores)
#cross validation of linear regreession model

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,

scoring="neg_mean_squared_error", cv=10)



lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
#Using random forest model



from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)



forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)



forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
#finding the best parametre



from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

]



forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error',

                           return_train_score=True)



grid_search.fit(housing_prepared, housing_labels)
#checkin best paramitre

grid_search.best_params_
#print all the combinations



cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
#cross validation of linear regreession model

grid_search_scores = cross_val_score(grid_search, housing_prepared, housing_labels,

                                     scoring="neg_mean_squared_error", cv=10)



grid_search_scores = np.sqrt(-lin_scores)

display_scores(grid_search_scores)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
#Testing the final model on test data set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)



final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2

display_scores(final_rmse)
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,

loc=squared_errors.mean(),

scale=stats.sem(squared_errors)))