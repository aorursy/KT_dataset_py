import pandas as pd

import sklearn

import numpy as np
cali_houses = pd.read_csv('../input/housing.csv')
cali_houses.head()
cali_houses.info()
cali_houses["ocean_proximity"].value_counts()
cali_houses.describe()
%matplotlib inline

import matplotlib.pyplot as plt 

cali_houses.hist(bins=40, figsize=(20,15))

plt.show()
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(cali_houses, test_size=0.2, random_state=34)
cali_houses["income_cat"] = np.ceil(cali_houses["median_income"] / 1.5)

cali_houses["income_cat"].where(cali_houses["income_cat"] < 5, 5.0, inplace=True)
cali_houses["income_cat"].hist(bins=20, figsize=(5,5))

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit 

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=34) 

for train_index, test_index in split.split(cali_houses, cali_houses["income_cat"]):

    strat_train_set = cali_houses.loc[train_index]

    strat_test_set = cali_houses.loc[test_index]
cali_houses["income_cat"].value_counts() / len(cali_houses)
for set_ in (strat_train_set, strat_test_set): 

    set_.drop("income_cat", axis=1, inplace=True)
cali_houses = strat_train_set.copy()
cali_houses.plot(kind="scatter", x="longitude", y="latitude")
cali_houses.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
cali_houses.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, 

             s=cali_houses["population"]/100, label="population", figsize=(12,8), 

             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,

)

plt.legend()
corr_matrix = cali_houses.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
import pandas.plotting
from pandas.plotting import scatter_matrix 

features = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(cali_houses[features], figsize=(12, 12))
cali_houses.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.5)
cali_houses["rooms_per_household"] = cali_houses["total_rooms"]/cali_houses["households"]

cali_houses["bedrooms_per_room"] = cali_houses["total_bedrooms"]/cali_houses["total_rooms"]

cali_houses["population_per_household"]=cali_houses["population"]/cali_houses["households"]
corr_matrix = cali_houses.corr() 

corr_matrix["median_house_value"].sort_values(ascending=False)
cali_houses = strat_train_set.drop("median_house_value", axis=1)

cali_houses_labels = strat_train_set["median_house_value"].copy()
sample_incomplete_rows = cali_houses[cali_houses.isnull().any(axis=1)].head()

sample_incomplete_rows
median = cali_houses["total_bedrooms"].median() 

cali_houses["total_bedrooms"].fillna(median, inplace=True)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
cali_houses_num =cali_houses.drop("ocean_proximity", axis=1)
imputer.fit(cali_houses_num)
imputer.statistics_
cali_houses_num.median().values
X = imputer.transform(cali_houses_num)
cali_houses_tr = pd.DataFrame(X, columns=cali_houses_num.columns,

                          index = list(cali_houses.index.values))

cali_houses_tr.loc[sample_incomplete_rows.index.values]
cali_houses_cat = cali_houses[['ocean_proximity']]

cali_houses_cat.head(10)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

cali_houses_cat_encoded = ordinal_encoder.fit_transform(cali_houses_cat)

cali_houses_cat_encoded[:10]
ordinal_encoder.categories_
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder() 

cali_houses_cat_1hot = cat_encoder.fit_transform(cali_houses_cat)

cali_houses_cat_1hot
cali_houses_cat_1hot.toarray()
cat_encoder = OneHotEncoder(sparse=False)

cali_houses_cat_1hot = cat_encoder.fit_transform(cali_houses_cat)

cali_houses_cat_1hot
cat_encoder.categories_
cali_houses.columns


from sklearn.base import BaseEstimator, TransformerMixin





rooms_ix, bedrooms_ix, population_ix, household_ix = [

    list(cali_houses.columns).index(col)

    for col in ("total_rooms", "total_bedrooms", "population", "households")]

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

cali_houses_extra_attribs = attr_adder.fit_transform(cali_houses.values)
cali_houses_extra_attribs = pd.DataFrame(

    cali_houses_extra_attribs,

    columns=list(cali_houses.columns)+["rooms_per_household", "population_per_household"])

cali_houses_extra_attribs.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),

        ('std_scaler', StandardScaler()),

    ])



cali_houses_num_tr = num_pipeline.fit_transform(cali_houses_num)
cali_houses_num_tr
from sklearn.compose import ColumnTransformer


num_attribs = list(cali_houses_num)

cat_attribs = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", OneHotEncoder(), cat_attribs),

    ])



cali_houses_prepared = full_pipeline.fit_transform(cali_houses)
cali_houses_prepared
cali_houses_prepared.shape
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(cali_houses_prepared, cali_houses_labels)


some_data = cali_houses.iloc[:5]

some_labels = cali_houses_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)



print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error



cali_houses_predictions = lin_reg.predict(cali_houses_prepared)

lin_mse = mean_squared_error(cali_houses_labels, cali_houses_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(cali_houses_prepared, cali_houses_labels)
cali_houses_predictions = tree_reg.predict(cali_houses_prepared)

tree_mse = mean_squared_error(cali_houses_labels, cali_houses_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse


from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, cali_houses_prepared, cali_houses_labels,

                         scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())



display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)

forest_reg.fit(cali_houses_prepared, cali_houses_labels)
cali_houses_predictions = forest_reg.predict(cali_houses_prepared)

forest_mse = mean_squared_error(cali_houses_labels, cali_houses_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
from sklearn.model_selection import cross_val_score



forest_scores = cross_val_score(forest_reg, cali_houses_prepared, cali_houses_labels,

                                scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV



param_grid = [

    

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

   

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestRegressor(random_state=42)



grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(cali_houses_prepared, cali_houses_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances


final_model = grid_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
final_rmse