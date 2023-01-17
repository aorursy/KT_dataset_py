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
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/california-housing-prices/housing.csv")
dataset.head()
dataset.info()
dataset['ocean_proximity'].value_counts()
dataset.describe()
#Lets plot whole dataset for a visual interpretation

%matplotlib inline

dataset.hist(bins=50, figsize=(20,18))
from sklearn.model_selection import train_test_split



train_set, test_set, = train_test_split(dataset, test_size = 0.2, random_state= 42)
dataset['income_cat'] = pd.cut(dataset['median_income'],

                                bins=[0.,1.5, 3.0, 4.5, 6., np.inf],

                                  labels=[1,2,3,4,5])
dataset['income_cat'].hist(figsize=(18,12))
#stratified sampling

from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits =1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(dataset, dataset['income_cat']):

    strat_train_set = dataset.iloc[train_index]

    strat_test_set = dataset.iloc[test_index]

    
strat_test_set['income_cat'].value_counts() / len(strat_test_set)
for set_ in (strat_test_set, strat_train_set):

    set_.drop("income_cat", axis=1, inplace=True)
dataset = strat_train_set.copy()
dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(18,10))
dataset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,

            s = dataset['population']/100, label="population", figsize=(14,8),

            c = "median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)

plt.legend()
corr_matrix  = dataset.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(dataset[attributes], figsize=(20,12))
dataset.plot(kind="scatter", x='median_income', y='median_house_value', alpha=0.2, figsize=(18,12))
dataset['rooms_per_household'] = dataset['total_rooms'] / dataset['households']

dataset['bedrooms_per_room'] = dataset['total_bedrooms'] / dataset['total_rooms']

dataset['population_per_household'] = dataset['population'] / dataset['households']
corr_matrix = dataset.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
dataset = strat_train_set.drop("median_house_value", axis=1)

dataset_labels = strat_train_set['median_house_value'].copy()
from sklearn.impute import SimpleImputer
dataset_num = dataset.drop("ocean_proximity", axis=1)
imputer = SimpleImputer(strategy="median")

imputer.fit(dataset_num)
imputer.statistics_
dataset_num.median().values
X = imputer.transform(dataset_num)
X
#Categorical Variable to NUMERICAL value

from sklearn.preprocessing import OrdinalEncoder

dataset_cat = dataset[['ocean_proximity']]

ordinal_encoder = OrdinalEncoder()

dataset_encoded = ordinal_encoder.fit_transform(dataset_cat)



dataset_encoded.shape
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

dataset_cat_1hot  = cat_encoder.fit_transform(dataset_cat)

dataset_cat_1hot
from sklearn.base import BaseEstimator, TransformerMixin



bedrooms_ix, population_ix, rooms_ix, households_ix = 3,4,5,6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self,add_bedrooms_per_room=True):

        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y = None):

        return self

    def transform(self, X, y=None):

        rooms_per_household = X[:, rooms_ix] /X[:,households_ix]

        population_per_houlsehold = X[:,population_ix]/ X[:, households_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]

            return np.c_[X, rooms_per_household, population_per_houlsehold, 

                        bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household,population_per_houlsehold]

        

    
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([

                            ('imputer', SimpleImputer(strategy='median')), 

                            ('attrib_adder',CombinedAttributesAdder()),

                            ('std_scaler', StandardScaler()) 

                        ])

                        
housing_num_tr = num_pipeline.fit_transform(dataset_num)
housing_num_tr
from sklearn.compose import ColumnTransformer

num_attribs = list(dataset_num)

cat_attribs = ['ocean_proximity']



full_pipeline= ColumnTransformer([("num", num_pipeline, num_attribs),

                                 ("cat", OneHotEncoder(), cat_attribs) ])
full_pipeline
housing_prepared = full_pipeline.fit_transform(dataset)
housing_prepared
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, dataset_labels)
some_data = dataset.iloc[:5]

some_labels = dataset_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels",list(some_labels))
#calculate error



from sklearn.metrics import mean_squared_error



housing_predictions = lin_reg.predict(housing_prepared)

lin_mse = mean_squared_error(dataset_labels, housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared, dataset_labels)
housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(dataset_labels, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score



scores = cross_val_score(tree_reg, housing_prepared, dataset_labels,

                        scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("Scores:", scores)

    print("Mean:", scores.mean() )

    print("Standard Deviation:", scores.std() )
display_scores(tree_rmse_scores)
lin_scores = cross_val_score(lin_reg, housing_prepared, dataset_labels,

                            scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



random_forest = RandomForestRegressor()

random_forest.fit(housing_prepared, dataset_labels)
random_forest_scores = cross_val_score(random_forest, housing_prepared, dataset_labels,

                                      scoring = "neg_mean_squared_error", cv=10)

random_forest_scores_sqrt = np.sqrt(-random_forest_scores)

display_scores(random_forest_scores_sqrt)
from sklearn.externals import joblib



joblib.dump(random_forest, "random_forest_model.pkl")

joblib.dump(lin_reg, "linear_regression_model.pkl")

joblib.dump(tree_reg, "tree_regression_model.pkl")
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV





param_grid = [

    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},

    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}

]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                          scoring="neg_mean_squared_error",

                          return_train_score=True)

grid_search.fit(housing_prepared, dataset_labels)





param_distributions = {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]}



forest_reg = RandomForestRegressor()

cv_ = RandomizedSearchCV(forest_reg, param_distributions, cv=5, 

                        scoring = "neg_mean_squared_error",

                        return_train_score=True)

cv_.fit
cv_.score
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres['params']):

    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_['cat']

cat_one_hot_encoder = list(cat_encoder.categories_[0])

attributes = num_attribs + extra_attribs + cat_one_hot_encoder

sorted(zip(feature_importances, attributes),reverse=True)
final_model = grid_search.best_estimator_



X_test = strat_test_set.drop("median_house_value", axis=1)

y_test = strat_test_set['median_house_value'].copy()



X_test_prepraded = full_pipeline.transform(X_test)



final_predictions = final_model.predict(X_test_prepraded)



joblib.dump(final_model, 'final_model.pkl')
testi_ = pd.DataFrame({'longitude':[-142.28], 'latitude':[42.12], 'housing_median_age':[29],

                      'total_rooms':[432],'total_bedrooms':[233],'population':[2000],'households':[2432],'median_income':[3.421],

                      'ocean_proximity':['INLAND']})
testi_
prepared_testi = full_pipeline.transform(testi_)
load_model = joblib.load('random_forest_model.pkl')

load_model2 = joblib.load('linear_regression_model.pkl')

load_model3 = joblib.load('tree_regression_model.pkl')

load_model4 = joblib.load('final_model.pkl')



print(load_model.predict(prepared_testi),

        load_model2.predict(prepared_testi),

        load_model3.predict(prepared_testi),

        load_model4.predict(prepared_testi)

     )
final_model.predict(prepared_testi)
X_test
final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse
X_test_prepraded.shape
dataset['ocean_proximity'].value_counts()
from sklearn import svm



svm_reg = svm.SVR(kernel="linear")

svm_reg.fit(housing_prepared, dataset_labels)

predict_smv_reg = svm_reg.predict(X_test_prepraded)







predict_smv_reg

final_mse_svm = mean_squared_error(y_test, predict_smv_reg)

final_rmse_smv = np.sqrt(final_mse_svm)

final_rmse_smv
svm_reg = svm.SVR(kernel="rbf")

svm_reg.fit(housing_prepared, dataset_labels)

predict_smv_reg = svm_reg.predict(X_test_prepraded)
predict_smv_reg

final_mse_svm = mean_squared_error(y_test, predict_smv_reg)

final_rmse_smv = np.sqrt(final_mse_svm)

final_rmse_smv
#Necesario un attributo para la prediccion.





X_test_prepraded.shape
cv_.fit(housing_prepared, dataset_labels)
full_pipeline_with_predictor = Pipeline([("preparation", full_pipeline),

        ("linear", LinearRegression())])
full_pipeline_with_predictor.fit(dataset, dataset_labels)

full_pipeline_with_predictor.predict(some_data)