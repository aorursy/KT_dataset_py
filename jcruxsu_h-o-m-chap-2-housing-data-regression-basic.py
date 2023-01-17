# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd

housing = pd.read_csv("../input/housing.csv")
housing.info()
housing.head()
housing.describe()
housing.hist(bins=50,figsize=(20,15))

plt.show()
housing["ocean_proximity"].value_counts()
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0,1.5,3,4.5,6,np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
from sklearn.model_selection import train_test_split

import sklearn
train_set,test_set = train_test_split(housing,test_size = 0.2,random_state=42)

for train_index,test_index in split.split(housing, housing["income_cat"]):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
strat_test_set["income_cat"].value_counts()
test_set["income_cat"].value_counts()
for set_ in (strat_train_set,strat_test_set):

    set_.drop("income_cat",axis=1,inplace=True)
strat_train_set.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4, s=housing["population"]/100, label="population",c="median_house_value",

              cmap= plt.get_cmap("jet"),colorbar=True)

plt.legend()
housing.plot(kind="scatter",x="median_income",y="median_house_value",alpha=0.2)
housing["rooms_per_house_hold"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
bed_median = housing["total_bedrooms"].median()

housing["total_bedrooms"].fillna(bed_median,inplace=True)
housing_labels = strat_train_set["median_house_value"].copy()

housing = strat_train_set.drop("median_house_value",axis=1)


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy ="median")
ocean_keys = dict(housing["ocean_proximity"].value_counts()).keys()
ocean_keys
housing["total_bedrooms"].fillna(bed_median,inplace=True)

housing_2= housing.copy()

INLAND_indexes = housing['ocean_proximity'].loc[lambda x: x=="INLAND"].index.tolist()

housing['ocean_proximity'].loc[INLAND_indexes] = 5.0



OCEAN_1H_indexes = housing['ocean_proximity'].loc[lambda x: x=="<1H OCEAN"].index.tolist()

housing['ocean_proximity'].loc[OCEAN_1H_indexes] = 4.0

NEAR_OCEAN_indexes =  housing['ocean_proximity'].loc[lambda x: x=="NEAR OCEAN"].index.tolist()

housing['ocean_proximity'].loc[NEAR_OCEAN_indexes] = 3.0

NEAR_BAY_indexes =  housing['ocean_proximity'].loc[lambda x: x=="NEAR BAY"].index.tolist()

housing['ocean_proximity'].loc[NEAR_BAY_indexes] = 2.0

ISLAND_indexes = housing['ocean_proximity'].loc[lambda x: x=="ISLAND"].index.tolist()

housing['ocean_proximity'].loc[ISLAND_indexes]=1.0





housing_num = housing.drop("ocean_proximity",axis=1)
housing["ocean_proximity"].value_counts()
imputer.fit(housing)

imputer_statistics_ = imputer.statistics_
housing.median().values

X = imputer.transform(housing)

housing_tr = pd.DataFrame(X,columns=housing.columns)
housing_tr
from sklearn.base import BaseEstimator,TransformerMixin



num_attributes=list(housing_num)

cat_attributes=['ocean_proximity']

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6





class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room 

    def fit(self, X,y=None):

        return self # nothing else to do 

    def transform(self, X,y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix] 

        population_per_household = X[:, population_ix] / X[:, household_ix] 

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix] 

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_2_encoded = ordinal_encoder.fit_transform(housing_2)
from sklearn.preprocessing import OneHotEncoder

two_encoder_one_hot =  OneHotEncoder()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline =  Pipeline([

    ('imputer',SimpleImputer(strategy="median")),

    ('attribs_adder',CombinedAttributesAdder()),

    ('std_scaler',StandardScaler()),

])
housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer



num_attribs =list(housing_num)

cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([

    ("num",num_pipeline,num_attribs),

    ("cat",OneHotEncoder(),cat_attribs)

])



housing_prepared = full_pipeline.fit_transform(housing)
housing
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared,housing_labels)
some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)
some_data
some_data_prepared
somedata_predicted = lin_reg.predict(some_data_prepared)
somedata_predicted
some_labels
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(housing_labels,housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor()

ada_reg.fit(housing_prepared,housing_labels)

housing_predictions = ada_reg.predict(housing_prepared)

ada_mse = mean_squared_error(housing_labels,housing_predictions)

ada_rmse = np.sqrt(ada_mse)

ada_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared,housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels,housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse

#overfit
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)

tree_rmse_scroes =np.sqrt(-scores)

tree_rmse_scroes



def show_results(scores):

    print("score",scores)

    print("score Mean : ",scores.mean())

    print("score deviation : ",scores.std())

    

show_results(tree_rmse_scroes)

    
lin_scores =  cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

show_results(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared,housing_labels)

forest_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels,housing_predictions)

forest_rmse = np.sqrt(forest_mse)

print(forest_rmse)



forest_scores = cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

show_results(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3,10,30],'max_features' : [2,4,6,8]},

    {'bootstrap':[False], 'n_estimators':[3,10],'max_features':[2,3,4]},

]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg,param_grid,cv=5,

                          scoring='neg_mean_squared_error',return_train_score=True)



grid_search.fit(housing_prepared,housing_labels)
grid_search.best_estimator_
grid_search.best_params_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):

    print(np.sqrt(-mean_score),params)
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value",axis=1)

Y_test = strat_test_set["median_house_value"].copy()
INLAND_indexes = X_test['ocean_proximity'].loc[lambda x: x=="INLAND"].index.tolist()

X_test['ocean_proximity'].loc[INLAND_indexes] = 5.0



OCEAN_1H_indexes = X_test['ocean_proximity'].loc[lambda x: x=="<1H OCEAN"].index.tolist()

X_test['ocean_proximity'].loc[OCEAN_1H_indexes] = 4.0

NEAR_OCEAN_indexes =  X_test['ocean_proximity'].loc[lambda x: x=="NEAR OCEAN"].index.tolist()

X_test['ocean_proximity'].loc[NEAR_OCEAN_indexes] = 3.0

NEAR_BAY_indexes =  X_test['ocean_proximity'].loc[lambda x: x=="NEAR BAY"].index.tolist()

X_test['ocean_proximity'].loc[NEAR_BAY_indexes] = 2.0

ISLAND_indexes = X_test['ocean_proximity'].loc[lambda x: x=="ISLAND"].index.tolist()

X_test['ocean_proximity'].loc[ISLAND_indexes]=1.0



x_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(x_test_prepared)

final_mse = mean_squared_error(Y_test,final_predictions)

final_rmse = np.sqrt(final_mse)

print(final_rmse)
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - Y_test) **2

np.sqrt(stats.t.interval(confidence,len(squared_errors)-1,loc = squared_errors.mean(),scale=stats.sem(squared_errors)))