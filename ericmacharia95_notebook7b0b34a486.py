import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import os
os.getcwd()
os.chdir('C:\\Users\\Mwaniki\\Desktop\\data science')
housing = pd.read_csv("C:\\Users\\Mwaniki\\Desktop\\data science\\housing.csv.csv")
housing.head()
housing.info()
housing.isnull().sum()
#columns have 20,640 with the total_bedroom having 207 missing entries.
housing["ocean_proximity"].value_counts()
housing.describe()
housing.hist(bins=100, figsize=(20,15))
plt.show()
#important features include: medium income.
housing["median_income"].hist()
# dividing medium income to limit number income category
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
#putting everything above 5th category as 5th category
housing['income_cat'].where(housing['income_cat'] < 5, other=5.0, inplace=True)
housing.head()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=29)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for items in (strat_train_set, strat_test_set):
    items.drop("income_cat", axis=1, inplace=True)
housing =  strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing['population']/100, label="population",
            figsize=(12,8), c="median_house_value", cmap=plt.get_cmap("jet"), sharex=False)
plt.legend()
import matplotlib.image as mpimg

ax = housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing['population']/100, label="population",
            figsize=(12,8), c="median_house_value", cmap=plt.get_cmap("jet"), sharex=False)
#load the png image
california_img = mpimg.imread("C:\\Users\\Mwaniki\\Desktop\\data science\\california.png")

plt.imshow(california_img, extent=[-124.55, -113.8, 32.4, 42.05], alpha=0.5, cmap=plt.get_cmap("jet"))

plt.xlabel("Longitude", fontsize=14)
plt.ylabel("Latitude", fontsize=14)
plt.legend(fontsize=14)
plt.show()
###looking for correlation(Pearson's Distance Correlation)

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending=False)
#feature engineering(adding more determinants)
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()
#Preparing the data for ML.

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#filling in missing data in total bedrooms

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
sample_incomplete_rows
#filling in values using sklearn

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)

imputer.fit(housing_num)
imputer.statistics_
housing_num.median().values
#using imputer to fill in missing values with median
X = imputer.transform(housing_num)


housing_tr = pd.DataFrame(X, columns=housing_num.columns)
housing_tr.isnull().values
housing_tr.head()
## handling categorical values using pandas

housing_cat = housing["ocean_proximity"]
housing_cat.head(20)
housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]
housing_categories
#dealing with categorical data using scikit learn

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(1, -1))
housing_cat_1hot
#onehot encoder returns sparse matrix use pandas to change to dense array

housing_cat_1hot.toarray()
housing.head()
from sklearn.base import BaseEstimator, TransformerMixin

#column indexes
rooms_ix, bedrooms_ix,population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self #nothing to return
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
    ("attribs_Adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr
class DataFrameSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, attribute_names):
        self.attibute_names = attribute_names
        
    def fit(self, X, y=None):
        return self #returns nothing
    
    def transform(self, X, y=None):
        return X[self.attibute_names].values
    
#complete pipeline

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
    ("cat_encoder", OneHotEncoder(sparse=False)),
])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])
housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
#selecting and training models

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#comparing prediction and actual values

some_data = housing.iloc[:5]
some_labels = housing_labels[:5]

some_data_prepared = full_pipeline.transform(some_data)
print("prediction: ", lin_reg.predict(some_data_prepared))
print("Actual Labels: ", list(some_labels))
from sklearn.metrics import mean_squared_error
housing_prediction = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels, housing_prediction)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

forest_reg = RandomForestRegressor(random_state=29)
forest_reg.fit(housing_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
housing_pred = forest_reg.predict(housing_prepared)

forest_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, cv=10, scoring="neg_mean_squared_error")

forest_rmse_scores = np.sqrt(-forest_scores)

def display_scores(scores):
    print("scores: ", scores)
    print("mean: ", scores.mean())
    print("std deviation: ", scores.std())

display_scores(forest_rmse_scores)
print("prediction: ", forest_reg.predict(housing_prepared)[:5])
print("Actual Labels: ", list(housing_labels)[:5])
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

rf_reg = RandomForestRegressor()

grid_search = GridSearchCV(rf_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)
rf_reg.get_params().keys()
#to get best combination of hyperparameters
grid_search.best_params_
#to get the best estimators directly
grid_search.best_estimator_
cv_res = grid_search.cv_results_

for mean_score, params in zip (cv_res["mean_test_score"], cv_res["params"]):
    print(np.sqrt(-mean_score), params)
