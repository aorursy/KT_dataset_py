import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots



from kaggle_tools import *



path = '../input/hands-on-machine-learning-housing-dataset/'

# Read data

housing = pd.read_csv(path+'housing.csv')
print(housing.info())

housing.head()
housing['ocean_proximity'].value_counts()
housing.describe()
mapboxpath = '../input/mapbox/mapbox.txt'

px.set_mapbox_access_token(open(mapboxpath).read())



fig = px.scatter_mapbox(housing, lat="latitude", lon="longitude",     color="median_house_value", size=housing["population"],

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=4.5)



fig.update_layout(title_text="California Housing Pricing (Median)")

fig.show()
check_num_col = housing.dtypes == 'float64'

num_col = sorted(check_num_col[check_num_col].index)

print('NUMERICAL COLUMNS: \n ', num_col)

print('HOW MANNY NUMERICAL COLUMNS? ', len(num_col))
dfplot = housing[num_col]



fig = make_subplots(rows=3, cols=3, subplot_titles=(num_col))



index = 0

for i in range(1,4):

    for j in range(1,4):      

        data = dfplot[num_col[index]]

        trace = go.Histogram(x=data, nbinsx=50)

        fig.append_trace(trace, i, j)

        index+=1

    

fig.update_layout(height=900, width=1250, title_text="Numerical Attributes")

fig.show()
from sklearn.model_selection import train_test_split

# Create a income category attribute

# More median income values are clustered around 1.5 to 6

# There are some median incomes beyond 6

bins_to_cut = [0., 1.5, 3.0, 4.5, 6., np.inf]

labels_to_cut = [1,2,3,4,5]

housing['income_cat'] = pd.cut(housing['median_income'], bins = bins_to_cut, labels=labels_to_cut )





fig = px.histogram(housing, x="income_cat")

fig.update_layout(height=500, width=700, title_text="Median Income Categories")

fig.show()
# Split into train and test using the stratified category

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)



for train_index, test_index in split.split(housing, housing['income_cat']):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

    

# Check distribution in the test and train sets

print('DISTRIBUTION TRAIN SET \n', strat_train_set['income_cat'].value_counts()/len(strat_train_set))

strat_train_set.drop('income_cat', axis=1, inplace=True)

print('-----------')

print('DISTRIBUTION TEST SET \n', strat_test_set['income_cat'].value_counts()/len(strat_test_set))

strat_test_set.drop('income_cat', axis=1, inplace=True)
corr = housing.corr()

corr["median_house_value"].sort_values(ascending=False)
fig = px.scatter(housing, x="median_income", y="median_house_value")

fig.show()
# Feature Engineering

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

housing["population_per_household"]=housing["population"]/housing["households"]



# Check again the correlation matrix

corr = housing.corr()

corr["median_house_value"].sort_values(ascending=False)
housing = strat_train_set.drop('median_house_value', axis=1)
# Numerical Attributes

housing_num = housing.drop('ocean_proximity', axis=1)

# Categorical Attributes

housing_cat = housing[['ocean_proximity']]

# Labels

housing_labels = strat_train_set['median_house_value'].copy()
# HANDLING NUMERICAL ATTRIBUTES

from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='median')

imputer.fit(housing_num)

X= imputer.transform(housing_num)



housing_tr = pd.DataFrame(X, columns=housing_num.columns)
# HANDLING CATEGORICAL ATTRIBUTES

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()



housing_cat_encoder = encoder.fit_transform(housing_cat)

print('CATEGORIES ENCODED:\n', encoder.categories_)
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]

# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]

# housing["population_per_household"]=housing["population"]/housing["households"]



from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):

        return self # nothing else to do

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]

        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]





attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)

housing_extra_attribs = attr_adder.transform(housing.values)
# Pipeline for numerical attributes

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



check_num_col = housing.dtypes == 'float64'

num_col = sorted(check_num_col[check_num_col].index)



pipeline_numeric = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('attribs_adder', CombinedAttributesAdder()),

    ('std_scaler', StandardScaler())

])

# Pipeline for categorical attributes

check_cat_col = housing.dtypes == 'object'

cat_col = sorted(check_cat_col[check_cat_col].index)

print('CAT COLS AFTER PROCESSED \n', cat_col)

# Full pipeline

from sklearn.compose import ColumnTransformer



full_pipeline = ColumnTransformer([

    ('pip_num', pipeline_numeric, num_col),

    ('pip_cat', OneHotEncoder(), cat_col)

])



housing_ready = full_pipeline.fit_transform(housing)

housing_ready
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression



def try_model(modelClass):

    model = modelClass

    model.fit(housing_ready, housing_labels)



    # Get the predictions

    predictors = housing.iloc[:5]

    labels = housing_labels.iloc[:5]



    predictors_tf = full_pipeline.transform(predictors)



    # See the predictions

    print('Predictions: \n', model.predict(predictors_tf))

    print('Labels: \n', list(labels))



    # Measure the error

    predictions_final = model.predict(housing_ready)

    rmse = np.sqrt(mean_squared_error(housing_labels, predictions_final))

    print('RMSE: ', round(rmse,2))

    

try_model(LinearRegression())
from sklearn.tree import DecisionTreeRegressor

try_model(DecisionTreeRegressor())
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor





def try_with_cv(modelClass, modelname):

    print('MODEL EVALUATION ', modelname)

    

    # Use model for calculating TRAINING ERROR

    modelClass.fit(housing_ready, housing_labels)

    housing_predictions = modelClass.predict(housing_ready)

    training_error = np.sqrt(mean_squared_error(housing_labels, housing_predictions))

    print('TRAINING ERROR: ', round(training_error,2),'\n')

    

    # Validation error using Cross Validation

    scores = cross_val_score(modelClass, housing_ready,housing_labels, scoring='neg_mean_squared_error', cv=10)

    scores_final = np.sqrt(-scores)

    

    print('VALIDATION ERROR:')

    print('RMSE(mean): ', scores_final.mean())

    print('RMSE(sd): ', scores_final.std())

    print('\n\n\n')

    

try_with_cv(DecisionTreeRegressor(), 'DECISION TREE REGRESSOR')

try_with_cv(RandomForestRegressor(), 'RANDOM FOREST REGRESSOR')
from sklearn.model_selection import GridSearchCV



# 90 rounds of training

# (3x4 + 2x3)x5

param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

]



forest_reg = RandomForestRegressor()

grid = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid.fit(housing_ready, housing_labels)
cvres = grid.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
final_model = grid.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)



final_predictions = final_model.predict(X_test_prepared)

final_mse = np.sqrt(mean_squared_error(y_test, final_predictions))

print('TEST SET MSE: ', final_mse)



features_weight = grid.best_estimator_.feature_importances_

custom_att = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["pip_cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])



attributes = num_col + custom_att + cat_one_hot_attribs



sorted(zip(features_weight, attributes), reverse=True)
from sklearn.svm import SVR

param_grid = [

        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},

        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],

         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},

    ]





grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)

grid_search.fit(housing_ready, housing_labels)
negative_mse = grid_search.best_score_

rmse = np.sqrt(-negative_mse)

print('SUPPORT VECTOR MACHINE RMSE: ', rmse)