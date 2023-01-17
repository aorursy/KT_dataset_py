

import os

import tarfile

import urllib



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"

HOUSING_PATH = os.path.join("datasets", "housing")

HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"



def fetch_housing_data (housing_url = HOUSING_URL, housing_path = HOUSING_PATH):

    

    os.makedirs(housing_path, exist_ok = True)

    tgz_path = os.path.join(housing_path, "housing.tgz")

    

    urllib.request.urlretrieve(housing_url, tgz_path)

    

    housing_tgz = tarfile.open(tgz_path)

    housing_tgz.extractall(housing_path)

    housing_tgz.close()
fetch_housing_data()
import pandas as pd



def load_housing_data(housing_path = HOUSING_PATH):

    

    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)
housing = load_housing_data()

housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
%matplotlib inline

import matplotlib.pyplot as plt



housing.hist(bins = 50, figsize = (25,20))

plt.show()
housing['median_income'].hist(figsize = (20,20))

plt.show()
import numpy as np



housing['income_category'] = pd.cut(housing['median_income'],

                                    bins = [0., 1.5, 3., 4.5, 6., np.inf],

                                    labels = [1, 2, 3, 4, 5])

housing['income_category'].value_counts()
housing['income_category'].hist()

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit as sss



def stratified_split(data, based_on, ratio = 0.2):

    split = sss(n_splits = 1, test_size = ratio, random_state = 42)



    for i, j in split.split(data, based_on):

        train = data.loc[i]

        test = data.loc[j]

        

    return train, test
train, test = stratified_split(housing, housing['income_category'])

test['income_category'].value_counts()/len(test)
housing['income_category'].value_counts()/len(housing)
train = train.drop(columns = ['income_category'])

test = test.drop(columns = ['income_category'])
housing = train.copy()
housing.plot(kind = "scatter", x = 'longitude', y = 'latitude', alpha = 0.1, figsize = (10, 7))

plt.show()

# alpha controls opacity based on density
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.4,

              s = housing['population']/100, c = 'median_house_value', cmap = plt.get_cmap('jet'),

              figsize = (10, 7), label = 'population', colorbar = True)

plt.legend()

plt.show()
corr = housing.corr()

corr['median_house_value'].sort_values(ascending = False)
from pandas.plotting import scatter_matrix



attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize = (12, 8))

plt.show()
housing.plot( kind = 'scatter', x = 'median_income', y = 'median_house_value', alpha = 0.25, figsize = (10,15))

plt.show()
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']

housing['population_per_household'] = housing['population'] / housing['households']



corr = housing.corr()

corr['median_house_value'].sort_values(ascending = False)
housing = train.drop(columns = ['median_house_value'])

labels = train['median_house_value'].copy()
from sklearn.impute import SimpleImputer as si



imputer = si(strategy = 'median')

housing_num = housing.drop(columns = ['ocean_proximity'])

imputer.fit(housing_num)
imputer.statistics_
median  = housing_num.median().values
x = imputer.transform(housing_num)

housing_tr = pd.DataFrame(x, columns = housing_num.columns, index = housing_num.index)

housing_tr.head()
from sklearn.preprocessing import OneHotEncoder



cat_encoder = OneHotEncoder()

housing_cat = housing[['ocean_proximity']]

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

housing_cat_1hot
from sklearn.base import BaseEstimator, TransformerMixin



room_i, bedroom_i, population_i, household_i = 3, 4, 5, 6



class combineAttributes(BaseEstimator, TransformerMixin):

    

    def __init__(self, _ = True):

        self._ = _

    

    def fit (self, x, y = None):

        return self

    

    def transform(self, x):

        

        rooms_per_household = x[:, room_i] / x[:, household_i]

        population_per_household = x[:, population_i] / x[:, household_i]

        bedrooms_per_room = x[:, bedroom_i] / x[:, room_i]

        

#         concatenate the new attributes to x and return 

        return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
attr_adder = combineAttributes(True)

housing_extra_attr = attr_adder.transform(housing.values)



housing_extra_attr = pd.DataFrame(

    housing_extra_attr,

    columns = list(housing.columns) + ['rooms_per_household', 'population_per_household', 'bedrooms_per_room'],

    index = housing.index)



housing_extra_attr.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('imputer', si(strategy = "median")),

    ('attribs_adder', combineAttributes()),

    ('std_scaler', StandardScaler())

])



housing_num_tr = num_pipeline.fit_transform(housing_num)

housing_num_tr
from sklearn.compose import ColumnTransformer



num_attr = list(housing_num)

cat_attr = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_attr),

    ('cat', OneHotEncoder(), cat_attr),

])



housing_prepd = full_pipeline.fit_transform(housing)

housing_prepd
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepd, labels)
eg_data = housing.iloc[:5]

eg_labels = labels.iloc[:5]

eg_prepd_data = full_pipeline.transform(eg_data)



print("Labels: ", list(eg_labels))

print("Predictions: ", lin_reg.predict(eg_prepd_data))
from sklearn.metrics import mean_squared_error



pred = lin_reg.predict(housing_prepd)

lin_mse = mean_squared_error(labels, pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state = 42)

tree_reg.fit(housing_prepd, labels)
pred = tree_reg.predict(housing_prepd)



tree_mse = mean_squared_error(labels, pred)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepd, labels,

                        scoring = 'neg_mean_squared_error', cv = 10)

tree_rmse_scores = np.sqrt( - scores)
def display_score(scores):

    print("Scores: ", scores)

    print("Mean: ", scores.mean())

    print("Standard Deviation: ", scores.std())

    

display_score(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepd, labels,

                            scoring = "neg_mean_squared_error", cv = 10)

lin_rmse_scores = np.sqrt(-lin_scores)



display_score(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

forest_reg.fit(housing_prepd, labels)
pred = forest_reg.predict(housing_prepd)



forest_scores = cross_val_score(forest_reg, housing_prepd, labels,

                               scoring = "neg_mean_squared_error", cv = 10)

forest_rmse_scores = np.sqrt(-forest_scores)

display_score(forest_rmse_scores)
def train_pred_cross_val(model, data, labels, ):

    

    model.fit(data, labels)

    pred = model.predict(data)

    scores = cross_val_score(model, data, labels,

                            scoring = "neg_mean_squared_error", cv = 10)

    rmse_scores = np.sqrt(- scores)

        

    return pred, rmse_scores
from sklearn.svm import SVR



svr_reg = SVR(kernel = "linear")



pred, svr_rmse_scores = train_pred_cross_val(svr_reg, housing_prepd, labels)

display_score(svr_rmse_scores)
from sklearn.model_selection import GridSearchCV



parameter_grid = [

    {'bootstrap' : [True], 'n_estimators' : [3, 10, 30], 'max_features' : [2, 4, 6, 8]},

    {'bootstrap' : [False], 'n_estimators' : [3, 10], 'max_features' : [2, 3, 4]}

]



grid_search = GridSearchCV(forest_reg, parameter_grid, cv = 5,

                           scoring = 'neg_mean_squared_error',

                           return_train_score = True)

grid_search.fit(housing_prepd, labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, parameters in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), parameters)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



parameter_distribution = {

    'n_estimators' : randint(1, 200),

    'max_features' : randint(1, 8)

}



forest_reg = RandomForestRegressor()

rnd_search = RandomizedSearchCV(forest_reg, param_distributions = parameter_distribution,

                                n_iter = 10, cv = 5, scoring = 'neg_mean_squared_error',

                                random_state = 42)



rnd_search.fit(housing_prepd, labels)
cvres = rnd_search.cv_results_



for mean_score, parameters in zip(cvres["mean_test_score"], cvres["params"]):

    print(parameters, np.sqrt(-mean_score))
feature_importance = grid_search.best_estimator_.feature_importances_

feature_importance   
extra_attr = ["rooms_per_household", "population_per_household", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attr = list(cat_encoder.categories_[0])

attributes = num_attr + extra_attr + cat_one_hot_attr

sorted(zip(feature_importance, attributes), reverse = True)
final_model = grid_search.best_estimator_



x_test = test.drop("median_house_value", axis = 1)

y_test = test["median_house_value"].copy()



x_test_prepd = full_pipeline.transform(x_test)

final_predictions = final_model.predict(x_test_prepd)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse