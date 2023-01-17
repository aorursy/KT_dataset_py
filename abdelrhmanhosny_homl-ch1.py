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
import seaborn as sns

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
housing = pd.read_csv('../input/california-housing-prices/housing.csv')
housing.head()
housing.info()
housing["ocean_proximity"].value_counts()
housing.describe()
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2 , random_state = 42)
housing["income_cat"] = pd.cut(housing["median_income"],[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(1,0.2,random_state = 42)
for train_ind , test_ind in split.split(housing,housing["income_cat"]):
    strat_train , strat_test = housing.loc[train_ind] , housing.loc[test_ind]
    
strat_train["income_cat"].value_counts()/len(strat_train)
for set_ in (strat_test,strat_train):
    set_.drop("income_cat",axis = 1,inplace = True)
plt.scatter(housing["longitude"],housing["latitude"])
plt.scatter(housing["longitude"],housing["latitude"],alpha=0.1)
plt.scatter(housing["longitude"],housing["latitude"],alpha=0.4,s=housing["population"]/100,
           c=housing["median_house_value"],cmap=plt.get_cmap("jet"))
plt.colorbar()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values()
from pandas.plotting import scatter_matrix

atts = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[atts],figsize=(20,16))
plt.plot(housing["median_income"],housing["median_house_value"],'o',alpha=0.1)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
plt.plot(housing["bedrooms_per_room"],housing["median_house_value"],'o',alpha=0.3)
housing = strat_train.drop("median_house_value",axis=1)
housing_labels = strat_train["median_house_value"].copy()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num);
imputer.statistics_
housing_num.median().values
X = imputer.transform(housing_num)
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(5)
housing_cat.value_counts()
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
from sklearn.base import BaseEstimator , TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room=False):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        rooms_per_household = X[:,rooms_ix]/X[:,households_ix]
        population_per_household = X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room :
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
             return np.c_[X,rooms_per_household,population_per_household]

attr_adder = CombinedAttributesAdder()
housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ('attribs_adder',CombinedAttributesAdder()),
    ('std_scaler',StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer

num_atts = list(housing_num)
cat_atts = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ('num',num_pipeline,num_atts),
    ('cat',OneHotEncoder(),cat_atts)
])

housing_prepared = full_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_pred = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_pred)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg,housing_prepared,housing_labels,
                        scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
scores = cross_val_score(forest_reg,housing_prepared,housing_labels,
                        scoring="neg_mean_squared_error",cv=3)
for_rmse_score= np.sqrt(-scores)
display_scores(for_rmse_score)
import joblib
joblib.dump(my_model,"mymodel.pkl")

#and retrieve by

my_loaded_model = joblib.load("mymodel.pkl")
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators':[3,10,30] , 'max_features':[2,4,6,8]},
              {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]}
             ]
forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg,param_grid,cv=3,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared,housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_atts + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
final_model = grid_search.best_estimator_

X_test = strat_test.drop("median_house_value",axis=1)
y_test = strat_test['median_house_value'].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
    loc=squared_errors.mean(),
    scale=stats.sem(squared_errors)))
