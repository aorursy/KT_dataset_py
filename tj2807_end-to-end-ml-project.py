# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
housing = pd.read_csv('../input/housing.csv')

housing.head()
housing.info()
housing.ocean_proximity.value_counts()
housing.describe()
housing.hist(bins = 50, figsize = (20,15))

plt.show()
from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing, test_size = 0.2, random_state=42)
housing['income_cat'] = np.ceil(housing.median_income/1.5)

housing.income_cat.where(housing.income_cat < 5, 5.0, inplace=True)

housing.income_cat.plot.hist()
# use scikit learn stratified split



from sklearn.model_selection import StratifiedShuffleSplit



splitObj = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in splitObj.split(housing, housing.income_cat):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]

(strat_test_set.income_cat.value_counts()/len(strat_test_set)).sort_index()
# drop added attribute



for set in (strat_train_set, strat_test_set):

    set.drop(['income_cat'], axis = 1, inplace = True)
housing = strat_train_set.copy()
housing.plot.scatter(x = 'longitude', y = 'latitude', alpha=0.4,

                    s = housing.population, label='population',

                    c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True, figsize = (60,60))

plt.legend()
housing.plot.hexbin(x = 'longitude', y='latitude', gridsize = 15)
# calculate correlation coefficient

# This might be important from feature engineering perspective, since two attributes which are heaviliy

# correlated may not be good as individual features



corr_matrix = housing.corr()

corr_matrix
# Pandas Scatter matrix can help plotting multiple scatter plots together



from pandas.tools.plotting import scatter_matrix



scatter_matrix(housing.loc[:,['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']].sample(1000),

              figsize = (12,8))
# Note that our target attribute is median_house_value. Most promising attribute to predict it seems to be

# median income with correlation of 0.68



housing.plot.scatter(x = 'median_income', y='median_house_value', alpha = 0.4, figsize = (30,30))
# Try combinations of features

housing['rooms_per_household'] = housing.total_rooms/housing.households

housing['bedrooms_per_room'] = housing.total_bedrooms / housing.total_rooms

housing['population_per_household'] = housing.population/housing.households
# Check if correlation has improved after attribute combinations

corrMat = housing.corr()

corrMat.median_house_value.sort_index()
# Drop labels from training set

housing = strat_train_set.drop('median_house_value', axis = 1)

housing_labels = strat_train_set['median_house_value'].copy()

housing.columns
# Missing Values

sample_missing_rows = housing[housing.isnull().any(axis=1)]

sample_missing_rows.head()
try:

    from sklearn.impute import SimpleImputer

except:

    from sklearn.preprocessing import Imputer as SimpleImputer



imputer = SimpleImputer(strategy = 'median')

# Removing text attribute since fit can't be done to text data.

housing_num = housing.drop('ocean_proximity', axis = 1)

housing_num.columns
imputer.fit(housing_num)

imputer.statistics_
# above should be same as

housing.median()
# transform the dataset, returns the numpy array

X = imputer.transform(housing_num)

X.shape
# convert to dataframe - Housing Truncated

housing_tr = pd.DataFrame(X, columns = housing_num.columns, index = housing.index)

housing_tr.head()
# See the filled values

housing_tr.loc[sample_missing_rows.index.values, :].head()
housing_cat = housing.ocean_proximity

housing_cat.head()
from sklearn.preprocessing import OrdinalEncoder





encoder = OrdinalEncoder()

housing_encoded = encoder.fit_transform(housing_cat.values.reshape(-1,1))

encoder.categories_



# This encoding is however problematic as model might learn these categories to be ordered, or more/less 

# important based on the number category is assigned. Model can also assume that two nearby values are more

# similar than distant values.
from sklearn.preprocessing import OneHotEncoder



one_hot_encoder = OneHotEncoder()

housing_onehot_encoded = one_hot_encoder.fit_transform(housing_cat.values.reshape(-1,1))

# above method returns a scipy sparse matrix, it can be converted to numpy dense array

housing_onehot_encoded.toarray()

# alternatively call OneHotEncoder(sparse=False)
# Custome tranformer can be implemented by adding fit, transform and fit_transform methods to a class.

# fit_tranform method, get_params and set_params method can be achieved by adding base classes

# BaseEstimator and TransformerMixin

from sklearn.base import BaseEstimator, TransformerMixin



# Writing a class for combined attributes adder

rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6



class CombinedAttributeAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room = True):

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, Y=None):

        return self

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix]/ X[:, household_ix]

        population_per_household = X[:, population_ix]/ X[:, household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:, bedrooms_ix]/ X[:, rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributeAdder(add_bedrooms_per_room = False)

housing_extra_array = attr_adder.fit_transform(housing.values)

housing_extra = pd.DataFrame(housing_extra_array, columns = list(housing.columns)+ 

             ['rooms_per_household', 'population_per_household'], index = housing.index)

# Optionally we can use FunctionTransformer to just convert a function into transformer.



# from sklearn.preprocessing import FunctionTransformer



# def add_extra_features(X, add_bedrooms_per_room=True):

#     rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

#     population_per_household = X[:, population_ix] / X[:, household_ix]

#     if add_bedrooms_per_room:

#         bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

#         return np.c_[X, rooms_per_household, population_per_household,

#                      bedrooms_per_room]

#     else:

#         return np.c_[X, rooms_per_household, population_per_household]



# attr_adder = FunctionTransformer(add_extra_features, validate=False,

#                                  kw_args={"add_bedrooms_per_room": False})

# housing_extra_attribs = attr_adder.fit_transform(housing.values)



housing.ocean_proximity.value_counts()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer



num_attribs = list(housing_num)

cat_attribs = ['ocean_proximity']



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy = 'median')),

    ('attributeAdder', CombinedAttributeAdder()),

    ('StandardScaler', StandardScaler())

])



full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_attribs),

   ('cat', OneHotEncoder(), cat_attribs)

])



housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared.shape

# Note that one hot encoded 5 categories take 5 columns, 3 for added attributes, 8 original columns.
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)



some_train_data = housing.iloc[:5]

some_train_labels = housing_labels.iloc[:5]

some_train_data_prepared = full_pipeline.transform(some_train_data)

some_train_data_predictions = lin_reg.predict(some_train_data_prepared)





results = pd.DataFrame({'labels': list(some_train_labels),'predictions': list(some_train_data_predictions)})

results['differencePercent'] = ((results.predictions - results.labels)*100)/results.labels

results.head()
from sklearn.metrics import mean_squared_error

predictions = lin_reg.predict(housing_prepared)

mse = mean_squared_error(predictions, housing_labels)

rmse = np.sqrt(mse)

rmse



# Average rmse of $68k is pretty bad.
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(predictions, housing_labels)

mae



# this model is clearly underfitting the data.

# one option would be feature engineering to add more valuable features. Another option is

# to try an use a more complex model.
# Decision tree regressor

from sklearn.tree import DecisionTreeRegressor



decTree = DecisionTreeRegressor()

decTree.fit(housing_prepared, housing_labels)

treePredictions = decTree.predict(housing_prepared)

rmse_tree = np.sqrt(mean_squared_error(treePredictions, housing_labels))

rmse_tree



# 0 error might mean model have overfit the data now.



# To confirm this we need a validation set. We can't use a test set since we might end up overfitting 

# the test set if we iterate our models using test set and then model may not become production ready.
# let's do 10 fold cross validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(decTree, housing_prepared, housing_labels,

                        scoring = 'neg_mean_squared_error', cv = 10)

def display_scores(scores):

    scores = np.sqrt(-scores)

    print(f"Scores: {scores}")

    print(f"Mean: {scores.mean()}")

    print(f"Standard Deviation: {scores.std()}")

    

display_scores(scores)
reg_scores = scores = cross_val_score(lin_reg, housing_prepared, housing_labels,

                        scoring = 'neg_mean_squared_error', cv = 10)

display_scores(reg_scores)
# train a RandomForestRegressor and check its performance

# Random forest trains decision trees on random subsets of features and then averages out their predictions

# called ensemble technique.

from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)

forest_predictions = forest_reg.predict(housing_prepared)

rmse = np.sqrt(mean_squared_error(forest_predictions, housing_labels))

rmse

# Random Forest is fitting the training dataset better than linear regression but worse than 

# decision tree. Let's check the validation set error
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,

                               scoring = 'neg_mean_squared_error', cv = 10)

display_scores(forest_scores)
# Trying out the SVM

from sklearn.svm import SVR

svm_reg = SVR(kernel='linear')

svm_reg.fit(housing_prepared, housing_labels)

svm_predictions = svm_reg.predict(housing_prepared)

rmse = np.sqrt(mean_squared_error(svm_predictions, housing_labels))

rmse
from sklearn.model_selection import GridSearchCV





param_grid = [

    # First try 12 combinations of features

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    # Then 6 combinations

    {'bootstrap': [False],'n_estimators': [3, 10],  'max_features': [2, 3, 4]}

]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)



# Total 90 combinations including cross validaiton
grid_search.best_params_
grid_search.best_estimator_

# Gives the best model 
# All the scores during the paramter search

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):

    print(np.sqrt(-mean_score), params)
pd.DataFrame(grid_search.cv_results_).head()
# Random Search

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



param_distribs = {

        'n_estimators': randint(low=1, high=200),

        'max_features': randint(low=1, high=8),

    }



forest_reg = RandomForestRegressor(random_state=42)

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,

                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
rnd_search.best_params_
# Get to know feature importances in random forest

feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution

cat_encoder = full_pipeline.named_transformers_["cat"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set["median_house_value"].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse
from scipy import stats



confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

mean = squared_errors.mean()

m = len(squared_errors)

# T Scores

np.sqrt(stats.t.interval(confidence, m - 1,

                         loc=np.mean(squared_errors),

                         scale=stats.sem(squared_errors)))
# T scores manual

tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)

tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)

np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)
# Z Score manually

zscore = stats.norm.ppf((1 + confidence) / 2)

zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)

np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
full_pipeline_with_predictor = Pipeline([

        ("preparation", full_pipeline),

        ("linear", LinearRegression())

    ])



# Transform and fit

full_pipeline_with_predictor.fit(housing, housing_labels)

# Transform and predict

full_pipeline_with_predictor.predict(X_test)
# Save the model

my_model = full_pipeline_with_predictor

from sklearn.externals import joblib



joblib.dump(my_model, "my_model.pkl") # DIFF

#...

my_model_loaded = joblib.load("my_model.pkl") # DIFF